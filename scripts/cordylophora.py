"""
Extraction + plotting for the Nandini et al. (2026) Cordylophora dataset.

Design goal: keep the *spreadsheet layout* declarative (one spec per sheet) and
the *parsing logic* generic. To adapt to a re-shaped workbook, edit the LAYOUT
constants only — never the extractor functions.

    Hoja1 -> aggregated functional response (mean +/- SE)        [Fig 2]
    Hoja3 -> raw preference counts: numbers AND biomass, 4 reps  [Figs 3 & 4]
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as pp
import seaborn as sb

# ----------------------------------------------------------------------------- #
#  Shared conventions
# ----------------------------------------------------------------------------- #
PREY_ORDER = ["rotifera", "harp", "cyclop"]          # Bp, Nitokra, Apocyclops
SAL_ORDER  = ["10gL", "20gL", "30gL"]
PREY_TITLE = {"rotifera": "B. plicatilis (rotifer)",
              "harp":     "N. lacustris (harpacticoid)",
              "cyclop":   "A. panamensis (cyclopoid)"}
PREY_PALETTE = {"rotifera": "#4C72B0", "harp": "#55A868", "cyclop": "#C44E52"}


def _read_sheet(path: str, sheet: str) -> pd.DataFrame:
    """Raw, header-less read. Single choke-point for the file I/O."""
    return pd.read_excel(path, sheet_name=sheet, header=None)


# ============================================================================= #
#  HOJA1  —  aggregated functional response
# ============================================================================= #
@dataclass(frozen=True)
class FRBlock:
    """Column layout for one prey type's block in Hoja1."""
    label: str                                   # tidy prey label
    density_col: int                             # column with offered density
    salinity_cols: Mapping[str, Tuple[int, int]] # salinity -> (mean_col, se_col)


# NB: the Bpli block stores salinities in the order 20/10/30 — declared here so
# the generic extractor never has to know about that quirk.
HOJA1_LAYOUT: Sequence[FRBlock] = (
    FRBlock("rotifera",  2, {"20gL": (3, 4),   "10gL": (5, 6),   "30gL": (7, 8)}),
    FRBlock("harp",     10, {"10gL": (11, 12), "20gL": (13, 14), "30gL": (15, 16)}),
    FRBlock("cyclop",   18, {"10gL": (19, 20), "20gL": (21, 22), "30gL": (23, 24)}),
)
HOJA1_DATA_ROWS = range(6, 13)   # 0-indexed rows holding the density levels


def _extract_fr_block(raw: pd.DataFrame, block: FRBlock,
                      data_rows=HOJA1_DATA_ROWS) -> list[dict]:
    """One prey block -> list of tidy long records. Skips empty cells."""
    out = []
    for r in data_rows:
        density = raw.iat[r, block.density_col]
        if pd.isna(density):
            continue
        for salinity, (mean_col, se_col) in block.salinity_cols.items():
            mean = raw.iat[r, mean_col]
            if pd.isna(mean):
                continue
            out.append({
                "prey_type":     block.label,
                "salinity":      salinity,
                "prey_density":  float(density),
                "mean_consumed": float(mean),
                "se_consumed":   float(raw.iat[r, se_col]),
            })
    return out


def extract_hoja1(path: str, sheet: str = "Hoja1",
                  layout: Sequence[FRBlock] = HOJA1_LAYOUT) -> pd.DataFrame:
    """Tidy long frame: prey_type, salinity, prey_density, mean_consumed, se_consumed."""
    raw = _read_sheet(path, sheet)
    rows: list[dict] = []
    for block in layout:
        rows.extend(_extract_fr_block(raw, block))
    df = pd.DataFrame(rows)
    df["prey_type"] = pd.Categorical(df["prey_type"], PREY_ORDER, ordered=True)
    df["salinity"]  = pd.Categorical(df["salinity"],  SAL_ORDER,  ordered=True)
    return df.sort_values(["prey_type", "salinity", "prey_density"]).reset_index(drop=True)


# --- optional Holling type II overlay (Y = a*X / (b + X)) -------------------- #
def _holling2(X, a, b):
    return a * X / (b + X)


def _fit_holling2(x, y):
    """Returns (a, b) or None if the fit fails. Isolated so it can be swapped."""
    from scipy.optimize import curve_fit
    try:
        (a, b), _ = curve_fit(_holling2, x, y, p0=[max(y) * 1.3, np.median(x)],
                              maxfev=10000, bounds=(0, np.inf))
        return a, b
    except Exception:
        return None


def plot_hoja1(df: pd.DataFrame, overlay_fit: bool = True,
               palette=PREY_PALETTE) -> pp.Figure:
    """3x3 grid (rows = salinity, cols = prey) of mean +/- SE consumption."""
    sb.set_theme(style="whitegrid", context="notebook")
    fig, axes = pp.subplots(len(SAL_ORDER), len(PREY_ORDER),
                            figsize=(12, 9), sharex="col", sharey="col")
    for i, sal in enumerate(SAL_ORDER):
        for j, prey in enumerate(PREY_ORDER):
            ax  = axes[i, j]
            sub = df[(df.salinity == sal) & (df.prey_type == prey)]
            color = palette[prey]
            ax.errorbar(sub.prey_density, sub.mean_consumed, yerr=sub.se_consumed,
                        fmt="o", color=color, capsize=3, markersize=6, lw=1.2, zorder=3)
            if overlay_fit and len(sub) >= 3:
                fit = _fit_holling2(sub.prey_density.values, sub.mean_consumed.values)
                if fit:
                    a, b = fit
                    xmax = sub.prey_density.max()
                    xs = np.linspace(0, xmax, 200)
                    ax.plot(xs, _holling2(xs, a, b), color=color, lw=1.6, alpha=.7, zorder=2)
                    # when no asymptote is reached, a & b run off to inf together
                    # (non-identified); only the initial slope a/b is meaningful.
                    if b > 5 * xmax:
                        ann = f"~linear\nslope a/b={a / b:.2f}"
                    else:
                        ann = f"a={a:.0f}\nb={b:.0f}"
                    ax.text(.96, .06, ann, transform=ax.transAxes,
                            ha="right", va="bottom", fontsize=8, color=color)
            if i == 0:
                ax.set_title(PREY_TITLE[prey], fontsize=10)
            if j == 0:
                ax.set_ylabel(f"{sal[:2]} g/L\nprey consumed", fontsize=9)
            if i == len(SAL_ORDER) - 1:
                ax.set_xlabel("prey density (offered, total)", fontsize=9)
    fig.suptitle("Hoja1 — functional response (mean ± SE, n=4)", y=1.0, fontsize=13)
    fig.tight_layout()
    return fig


# ============================================================================= #
#  HOJA3  —  raw preference counts (numbers + biomass)
# ============================================================================= #
@dataclass(frozen=True)
class PrefPair:
    """Column layout for one prey type in Hoja3 (paired number/biomass cols)."""
    label: str
    salinity_cols: Mapping[str, Tuple[int, int]]  # salinity -> (number_col, biomass_col)


HOJA3_LAYOUT: Sequence[PrefPair] = (
    PrefPair("cyclop",   {"10gL": (0, 1),   "20gL": (2, 3),   "30gL": (4, 5)}),    # Apo
    PrefPair("harp",     {"10gL": (7, 8),   "20gL": (9, 10),  "30gL": (11, 12)}),  # Nito
    PrefPair("rotifera", {"10gL": (13, 14), "20gL": (15, 16), "30gL": (17, 18)}),  # Bp
)
HOJA3_REP_ROWS = range(2, 6)   # 4 replicate rows
HOJA3_OFFERED  = {"cyclop": 20, "harp": 20, "rotifera": 50}   # prey offered per type


def _extract_pref_block(raw: pd.DataFrame, pair: PrefPair,
                        rep_rows=HOJA3_REP_ROWS) -> list[dict]:
    out = []
    for rep, r in enumerate(rep_rows):
        for salinity, (num_col, bio_col) in pair.salinity_cols.items():
            number = raw.iat[r, num_col]
            if pd.isna(number):
                continue
            out.append({
                "prey_type": pair.label,
                "salinity":  salinity,
                "replicate": rep + 1,
                "number":    float(number),
                "biomass":   float(raw.iat[r, bio_col]),
            })
    return out


def extract_hoja3(path: str, sheet: str = "Hoja3",
                  layout: Sequence[PrefPair] = HOJA3_LAYOUT) -> pd.DataFrame:
    """Replicate-level long frame: prey_type, salinity, replicate, number, biomass."""
    raw = _read_sheet(path, sheet)
    rows: list[dict] = []
    for pair in layout:
        rows.extend(_extract_pref_block(raw, pair))
    df = pd.DataFrame(rows)
    df["prey_type"] = pd.Categorical(df["prey_type"], PREY_ORDER, ordered=True)
    df["salinity"]  = pd.Categorical(df["salinity"],  SAL_ORDER,  ordered=True)
    return df.sort_values(["prey_type", "salinity", "replicate"]).reset_index(drop=True)


def summarize_hoja3(df: pd.DataFrame, metric: str = "number") -> pd.DataFrame:
    """mean + SE (ddof=1) per prey_type x salinity for the chosen metric."""
    g = df.groupby(["prey_type", "salinity"], observed=True)[metric]
    return g.agg(mean="mean", se=lambda x: x.std(ddof=1) / np.sqrt(x.count())).reset_index()


def plot_hoja3(df: pd.DataFrame, metric: str = "number",
               palette=PREY_PALETTE) -> pp.Figure:
    """Grouped bars (mean +/- SE from replicates) — `metric` is 'number' or 'biomass'."""
    sb.set_theme(style="whitegrid", context="notebook")
    label = {"number": "prey consumed (count)",
             "biomass": "prey consumed (biomass, µg)"}[metric]
    fig, ax = pp.subplots(figsize=(8, 5))
    sb.barplot(df, x="salinity", y=metric, hue="prey_type",
               order=SAL_ORDER, hue_order=PREY_ORDER, palette=palette,
               errorbar=("se", 1), capsize=.08, err_kws={"linewidth": 1.2}, ax=ax)
    ax.set_xlabel("salinity"); ax.set_ylabel(label)
    ax.set_title(f"Hoja3 — preference assay, {metric} (mean ± SE, n=4)")
    ax.legend(title="prey", labels=[PREY_TITLE[p] for p in PREY_ORDER], fontsize=8)
    fig.tight_layout()
    return fig


# ============================================================================= #
if __name__ == "__main__":
    PATH = "/mnt/user-data/uploads/dataset.xlsx"

    h1 = extract_hoja1(PATH)
    print("Hoja1:", h1.shape)
    plot_hoja1(h1).savefig("/home/claude/fig_hoja1.png", dpi=130, bbox_inches="tight")

    h3 = extract_hoja3(PATH)
    print("Hoja3:", h3.shape)
    print(summarize_hoja3(h3, "number").to_string(index=False))
    plot_hoja3(h3, "number").savefig("/home/claude/fig_hoja3_number.png", dpi=130, bbox_inches="tight")
    plot_hoja3(h3, "biomass").savefig("/home/claude/fig_hoja3_biomass.png", dpi=130, bbox_inches="tight")
