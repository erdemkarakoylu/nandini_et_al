# Estimate, Don't Test. Causal Bayesian Inference in Marine Ecology

A causal-graph-first, Bayesian reanalysis of the functional response of the invasive
hydroid *Cordylophora caspia*, reanalyzing the dataset of Nandini et al. (2026) as a
worked demonstration of an alternative to null-hypothesis significance testing (NHST)
for controlled ecological experiments. Part I of a two-paper series; a companion paper
extending the same graph-first workflow to observational data is planned separately.

## Why this repository exists

The original study drew its conclusions from two-way ANOVA on prey numbers and, in a
separate analysis, prey biomass. The two analyses disagree: salinity and a
prey$\times$salinity interaction are significant for numbers consumed, but not for
biomass — and the paper's summary conclusion ("salinity had no effect") reflects only
the biomass result. This is not presented as an error specific to that paper; it is an
illustration of a general property of NHST — a non-significant result is easily, and
commonly, generalized into "no effect," a documented and worsening pattern in ecological
reporting.

This repository reanalyzes the same dataset with a different discipline: draw the causal
graph before writing any statistical model, fit a Bayesian generative model consistent
with that graph, elicit every prior from background knowledge stated in advance (never
from the data being analyzed), and report every conclusion as a posterior effect size
with an interval, checked for sensitivity to the prior actually chosen — reporting
whichever answer that check returns, not the answer that was hoped for.

## Modeling summary

The functional response is modeled as Holling's type II disc equation,

$$\mu = \frac{\alpha N_0}{1 + \alpha h N_0}$$

with attack rate $\alpha$ (logit link, bounded in $(0,1)$) and handling time $h$ (log
link, positive), each carrying a `{prey, salinity, prey$\times$salinity interaction}`
structure. Main effects and the interaction are parameterized as `ZeroSumNormal`: no
prey type or salinity level is a privileged reference, so every coefficient is a
symmetric deviation from a shared grand mean, constrained to sum to zero across levels.
Priors on every term are elicited in `01_prior_elicitation.ipynb`, entirely without
reference to the dataset: each is built from an interpretable, literature-anchored claim
(a capture fraction, a fold-change between levels) mapped to its maximum-entropy
distribution.

Two measurement-layer variants are built and compared:

- **M1 (primary, reported model).** A gamma likelihood on the raw count $\bar y$, with
  scale $\sigma_{\text{eff}} = \sqrt{\tau^2 + se^2}$ combining an estimated
  process/lack-of-fit term $\tau$ with the reported standard error $se$ in quadrature.
- **M2 (compared alternative).** A beta likelihood on the fraction consumed
  $\bar y / N_0$, bounded in $(0,1)$ by construction — addressing a small, physically
  impossible ceiling violation in M1's prior predictive distribution (simulated
  consumption occasionally exceeding the offered density, since the gamma's support is
  unbounded even though its mean is not).

The two are compared by leave-one-out cross-validation, with an explicit Jacobian
correction for the change of variables between the count and proportion scales (see
`manuscript/supplementary.qmd`). M1 is preferred by a wide margin and is the model
reported throughout the paper; M2's fit, comparison, and remaining open items (its
dispersion parameter is not yet elicited) are documented in the supplementary material
rather than adopted.

Every parameter flagged by power-scaling prior sensitivity (`psense`) is followed up
with a prior-width sweep rather than reported at face value. The outcomes differ: the
salinity effect on attack rate is directionally stable across a wide range of prior
choices and is reported as a finding; the prey$\times$salinity interaction's apparent
significance depends on adopting a prior looser than the one actually elicited, and is
**not** reported as a confirmed effect. Both outcomes are treated as legitimate results
of the same check, not selectively reported.

No prey-preference or choice model is built in this repository. The raw preference
counts (`Hoja3` in the authors' data) are extracted and available
(`hoja3_prey_preference.csv`) but not yet modeled; this is future work, not a completed
phase.

## Repository structure

```text
.
├── .gitignore
├── data/
│   ├── sent_data/                           # NOT TRACKED -- see "Data availability and licensing" below
│   │   ├── dataset.xlsx                     # raw data sent by the study's authors
│   │   └── processed/
│   │       ├── hoja1_functional_response.csv  # aggregated FR data (Figure 2), authoritative
│   │       └── hoja3_prey_preference.csv      # raw preference-assay replicate counts
│   └── extracted_data/                      # tracked -- digitized from a CC-BY-licensed figure
│       ├── figure_2/                        # hand-digitized points from the original Figure 2
│       └── processed/
│           └── d_final.csv                  # digitized reconstruction; not what M1/M2 were fit to, but the dataset publicly available here (see "Data availability and licensing")
├── notebooks/
│   ├── 00_raw_data_wrangle.ipynb            # authoritative extraction: Hoja1 + Hoja3 from dataset.xlsx
│   ├── 00_digitized_data_wrangle.ipynb      # digitized-from-figure reconstruction + cross-check (run after 00_raw_*)
│   ├── 01_prior_elicitation.ipynb           # all priors, data-free by design
│   └── 02_sampling_and_inference.ipynb      # M1 + M2 build, sampling, diagnostics, contrasts, prior sensitivity, LOO comparison
├── scripts/
│   ├── model.py                             # build_model (M1), build_model_m2 (M2)
│   ├── plot_utils.py                        # DAG rendering, forest plots, prior-vs-posterior plots
│   └── stats_utils.py                       # posterior reconstruction, contrasts, calibration summaries
├── figures/                                  # NOT TRACKED -- regenerable; all rendered manuscript figures
├── artifacts/                                 # NOT TRACKED -- saved InferenceData checkpoints (e.g. m1_idata.nc)
└── manuscript/
    ├── main.qmd                             # top-level Quarto document (includes the sections below)
    ├── introduction.qmd
    ├── methods.qmd
    ├── results.qmd
    ├── discussion.qmd
    ├── supplementary.qmd                    # M2 model spec and the M1-vs-M2 comparison (not included in main.qmd by default)
    └── references.bib
```

## Data availability and licensing

Two data sources feed this repository, with different licensing status, and they are
treated differently as a result.

**`data/extracted_data/` — tracked.** Digitized directly from the original paper's
published Figure 2. Nandini et al. (2026) is published under CC-BY 4.0, which
explicitly permits reuse, distribution, and reproduction in any medium provided the
original work is cited — so redistributing points extracted from that published figure
is within the terms of the license the authors themselves chose.

**`data/sent_data/` — not tracked.** The raw Excel file behind this reanalysis was
shared directly by the study's authors, outside of any formal publication or open-data
deposit. The paper's own Data Availability statement reads "data will be made available
on reasonable request" — a private, gatekept arrangement, not an open license.
Agreeing to share the file with one requester for reanalysis is not the same as
consenting to its public redistribution, so it is not included here, and neither are
the CSVs derived directly from it (`hoja1_functional_response.csv`,
`hoja3_prey_preference.csv`), regardless of how closely their values match the
CC-BY-derived digitized reconstruction (`d_final.csv`) -- the two were cross-validated
at r = 0.9999 in `00_digitized_data_wrangle.ipynb`, but that numerical similarity does
not transfer redistribution rights from one source to the other.

**Practical consequence:** a fresh clone of this repository cannot run
`00_raw_data_wrangle.ipynb`, or anything downstream of its output (including the
primary model in `02_sampling_and_inference.ipynb`), without separately obtaining
`dataset.xlsx`. To request it, see the corresponding author's contact details in
Nandini et al. (2026), or open an issue in this repository.

**What this means for citing this dataset.** `d_final.csv` and the raw digitized points
in `data/extracted_data/figure_2/` are the only data files this repository can
reasonably make public, and they are accordingly what its own data-availability statement
points to. This is a deliberate scope limitation, not an oversight: the manuscript's
data-availability statement reads

> The raw experimental data underlying this reanalysis were provided directly by the
> authors of the original study (Nandini et al. 2026,
> https://doi.org/10.1093/plankt/fbaf065) and are available from them on reasonable
> request, per that paper's own data availability statement. A digitized
> reconstruction derived from that paper's openly licensed (CC-BY 4.0) Figure 2 --
> used here only as an independent cross-check on the primary extraction, not as the
> data the reported models were fit to -- is version-controlled in this repository at
> `data/extracted_data/`.

The distinction to hold onto: `hoja1_functional_response.csv` (gated, from the
authors) is what M1 and M2 are actually fit to and what every number in the manuscript
comes from. `d_final.csv` (public, digitized) is a validation artifact that happens to
track it closely (r = 0.9999) -- not an interchangeable substitute, and not what
"available in this repository" should be read as endorsing as the primary source.

## Reproducing this analysis

1. `00_raw_data_wrangle.ipynb` — must run first; produces the CSVs everything else
   depends on. **Requires `data/sent_data/dataset.xlsx`**, which is not included in
   this repository (see "Data availability and licensing" above) and must be obtained
   separately before this step will run.
2. `00_digitized_data_wrangle.ipynb` — optional cross-check against (1)'s output;
   depends on it despite the alphabetical filename order. The digitized source data
   (`data/extracted_data/`) is included, but the comparison step still needs (1)'s
   output to compare against.
3. `01_prior_elicitation.ipynb` — no dependency on (1) or (2), and no dependency on
   any data file; can run any time.
4. `02_sampling_and_inference.ipynb` — depends on (1)'s output and on `scripts/model.py`.
   That said minimal modification could have the model using the digitized data from (2) instead of the original data from (1).

To render the manuscript, run Quarto on `manuscript/main.qmd`. `supplementary.qmd` is
not included by default; add `{{< include supplementary.qmd >}}` to `main.qmd` to
include the M2 model and comparison in the rendered output.

For the extraction logic and known layout quirks of the source spreadsheet, see
`00_raw_data_wrangle.ipynb`; for the experimental design, see the manuscript's
Data and study system section (`manuscript/methods.qmd`).
