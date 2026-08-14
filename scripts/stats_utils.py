"""Posterior post-processing helpers for the *Cordylophora* functional-
response model.

Contains functions to reconstruct the mechanistic parameters (attack rate,
handling time) on the full prey-by-salinity grid from the fitted effect
coefficients, compute expected consumption at a given offered density, and
summarize pairwise posterior contrasts (mean, 94% HDI, and directional
probability) between grid cells.
"""
import numpy as np
import pandas as pd
import xarray as xr
import arviz_stats as azs


def reconstruct_mechanistic_parameters(posterior):
    """Reconstruct attack rate and handling time on the full (prey,
    salinity) grid from the posterior's effect coefficients.

    Rebuilding on the grid (rather than reading the per-observation
    ``\u03b1``/``h`` stored by the likelihood) decouples contrasts from
    which densities happen to have been tested in a given cell.

    Parameters
    ----------
    posterior : xarray.Dataset
        The ``.posterior`` group of a fitted model's InferenceData,
        containing ``a0``, ``a_prey``, ``a_sal``, ``a_int``, ``h0``,
        ``h_prey``, ``h_sal``, ``h_int``.

    Returns
    -------
    attack_rate : xarray.DataArray
        Attack rate, dims ``(chain, draw, prey, salinity)``, in (0, 1).
    handling_time : xarray.DataArray
        Handling time, dims ``(chain, draw, prey, salinity)``, > 0.
    """
    attack_rate = 1 / (
        1
        + np.exp(
            -(
                posterior["a0"]
                + posterior["a_prey"]
                + posterior["a_sal"]
                + posterior["a_int"]
            )
        )
    )
    handling_time = np.exp(
        posterior["h0"] + posterior["h_prey"] + posterior["h_sal"] + posterior["h_int"]
    )
    return attack_rate, handling_time


def expected_eaten(attack_rate, handling_time, n_offered):
    """Expected number of prey removed at a given offered density, per the
    Holling type-II disc equation.

    Parameters
    ----------
    attack_rate : xarray.DataArray
        Attack rate, as returned by :func:`reconstruct_mechanistic_parameters`.
    handling_time : xarray.DataArray
        Handling time, as returned by :func:`reconstruct_mechanistic_parameters`.
    n_offered : float
        Offered prey density at which to evaluate expected consumption.

    Returns
    -------
    xarray.DataArray
        Expected number of prey removed, same dims as ``attack_rate``.
    """
    return (
        attack_rate
        * n_offered
        / (1 + attack_rate * handling_time * n_offered)
    )


def compare(a, b, label, prob=0.94):
    """Print the posterior contrast ``a - b``: mean, HDI, and P(a > b).

    Parameters
    ----------
    a, b : xarray.DataArray
        Posterior quantities to contrast, each with ``(chain, draw)``
        dims (plus any remaining shared dims, e.g. after ``.sel()``/
        ``.mean()`` down to a single series).
    label : str
        Text label printed alongside the summary.
    prob : float, default 0.94
        HDI probability mass.

    Returns
    -------
    None
        Prints the summary; does not return a value (matches the notebook
        cell this was extracted from).
    """
    diff = a - b  # keep as DataArray with (chain, draw) dims
    hdi = azs.hdi(diff, prob=prob)
    lo, hi = float(hdi.sel(ci_bound="lower")), float(hdi.sel(ci_bound="upper"))
    p_gt = float((diff.stack(z=("chain", "draw")) > 0).mean())
    print(
        f"{label:40s} mean={float(diff.mean()):+.3f}  "
        f"{int(prob * 100)}% HDI=[{lo:+.3f}, {hi:+.3f}]  P(a>b)={p_gt:.2f}"
    )


def summarize_ppc(idata, var_name="likelihood", group="posterior_predictive", prob=0.94):
    """Numeric summary of a posterior (or prior) predictive check: the
    textual counterpart to :func:`plot_utils` PPC plots.

    Reports the fraction of observed values falling inside their own
    per-observation predictive HDI ("coverage" — should be close to `prob`
    for a well-calibrated model) plus mean absolute and mean signed error
    of the predictive mean against the observed value.

    Parameters
    ----------
    idata : xarray.DataTree
        InferenceData-like object with an ``observed_data`` group and a
        ``group`` (default ``"posterior_predictive"``) group, both
        containing ``var_name``.
    var_name : str, default "likelihood"
        Name of the observed/predicted variable.
    group : str, default "posterior_predictive"
        Which predictive group to summarize (e.g. ``"prior_predictive"``).
    prob : float, default 0.94
        HDI probability mass used for the coverage check.

    Returns
    -------
    dict
        Keys: ``coverage`` (fraction of observations inside their HDI),
        ``prob`` (the HDI mass used), ``mean_abs_error``,
        ``mean_signed_error`` (predictive mean minus observed, so positive
        means the model over-predicts on average).
    """
    pred = idata[group][var_name]
    obs = idata.observed_data[var_name]
    hdi = azs.hdi(pred, prob=prob)
    lo, hi = hdi.sel(ci_bound="lower"), hdi.sel(ci_bound="upper")
    coverage = float(((obs >= lo) & (obs <= hi)).mean())
    pred_mean = pred.mean(("chain", "draw"))
    mae = float(np.abs(pred_mean - obs).mean())
    mse = float((pred_mean - obs).mean())
    return {
        "coverage": coverage, "prob": prob,
        "mean_abs_error": mae, "mean_signed_error": mse,
    }


def summarize_contrasts(pieces, prob=0.94, ref=0.0):
    """Tabular summary of named posterior contrasts: the textual
    counterpart to a :func:`plot_utils.render_forest` figure.

    Parameters
    ----------
    pieces : dict of str -> xarray.DataArray
        Mapping from contrast label to posterior contrast, as passed to
        :func:`named_contrast_dataset`.
    prob : float, default 0.94
        HDI probability mass.
    ref : float, default 0.0
        Reference ("no effect") value — 0 for a difference contrast, 1 for
        a ratio contrast. Used to compute the directional probability
        ``P(contrast > ref)``.

    Returns
    -------
    pandas.DataFrame
        Columns: ``contrast``, ``mean``, ``hdi_low``, ``hdi_high``,
        ``p_gt_ref``, indexed by contrast label.
    """
    rows = []
    for label, d in pieces.items():
        hdi = azs.hdi(d, prob=prob)
        lo, hi = float(hdi.sel(ci_bound="lower")), float(hdi.sel(ci_bound="upper"))
        p_gt = float((d.stack(z=("chain", "draw")) > ref).mean())
        rows.append({
            "contrast": label, "mean": float(d.mean()),
            "hdi_low": lo, "hdi_high": hi, "p_gt_ref": p_gt,
        })
    return pd.DataFrame(rows).set_index("contrast")


def summarize_prior_posterior(idata, var_names, prob=0.94):
    """Tabular summary of prior-vs-posterior shift: the textual
    counterpart to a :func:`plot_utils.render_prior_posterior` figure.

    For each named variable, reports the prior and posterior mean/SD, the
    posterior shift expressed in prior-SD units (how many prior standard
    deviations the mean moved), and the SD ratio (posterior SD / prior
    SD — well below 1 indicates the data meaningfully narrowed the prior;
    close to 1 indicates the posterior is still mostly prior-driven).

    Parameters
    ----------
    idata : xarray.DataTree
        InferenceData-like object with both ``prior`` and ``posterior``
        groups.
    var_names : list of str
        Variables to summarize. Multi-dimensional variables (e.g. with a
        ``prey`` or ``salinity`` dim) are summarized per level.
    prob : float, default 0.94
        Unused directly (kept for interface symmetry with the other
        summary functions); shift and SD ratio are reported instead of an
        HDI, since the comparison of interest here is prior vs. posterior
        spread, not a single interval.

    Returns
    -------
    pandas.DataFrame
        Columns: ``variable``, ``prior_mean``, ``prior_sd``,
        ``post_mean``, ``post_sd``, ``shift_in_prior_sd``, ``sd_ratio``.
    """
    rows = []
    for name in var_names:
        prior = idata.prior[name]
        post = idata.posterior[name]
        extra_dims = [d for d in prior.dims if d not in ("chain", "draw")]
        if not extra_dims:
            combos = [{}]
        else:
            import itertools
            levels = [prior.coords[d].values for d in extra_dims]
            combos = [dict(zip(extra_dims, vals)) for vals in itertools.product(*levels)]
        for combo in combos:
            pr = prior.sel(combo) if combo else prior
            po = post.sel(combo) if combo else post
            pr_mean, pr_sd = float(pr.mean()), float(pr.std())
            po_mean, po_sd = float(po.mean()), float(po.std())
            label = name if not combo else f"{name}[{','.join(str(v) for v in combo.values())}]"
            rows.append({
                "variable": label, "prior_mean": pr_mean, "prior_sd": pr_sd,
                "post_mean": po_mean, "post_sd": po_sd,
                "shift_in_prior_sd": (po_mean - pr_mean) / pr_sd if pr_sd else np.nan,
                "sd_ratio": po_sd / pr_sd if pr_sd else np.nan,
            })
    return pd.DataFrame(rows).set_index("variable")


def named_contrast_dataset(pieces, var_name):
    """Assemble a dict of named posterior contrasts into a single labeled
    xarray Dataset, ready for :func:`plot_utils.render_forest`.

    Parameters
    ----------
    pieces : dict of str -> xarray.DataArray
        Mapping from a human-readable contrast label (e.g. ``"Nito - Apo"``)
        to the corresponding posterior contrast DataArray.
    var_name : str
        Name to give the single data variable in the resulting Dataset
        (used as the plot's variable label).

    Returns
    -------
    xarray.Dataset
        Dataset with one data variable (``var_name``) and a ``contrast``
        dimension carrying the dict keys as string coordinates.
    """
    contrasts = xr.concat(
        list(pieces.values()),
        dim=xr.DataArray(list(pieces.keys()), dims="contrast", name="contrast"),
    )
    return contrasts.to_dataset(name=var_name)
