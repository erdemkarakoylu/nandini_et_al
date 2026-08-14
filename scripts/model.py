"""PyMC model specification for the *Cordylophora caspia* functional-response
re-analysis.

This module contains the single model currently fit to the data (referred to
throughout the project as **M1**): a Holling type-II disc equation with
attack rate and handling time each carrying a {prey, salinity, interaction}
structure, and a two-component (process + measurement) Gamma likelihood.

An earlier structural baseline (M0, known-SE-only likelihood) was used only
to confirm the mechanistic structure sampled cleanly before the process-error
term was added, and is not retained here. A bounded-support alternative
(M2, Beta likelihood on the proportion consumed) is planned but not yet
implemented — see the project notebooks for the rationale.

Notes
-----
Priors are the maximum-entropy values derived in
``01_prior_elicitation.ipynb`` from interpretable, pre-data claims (see that
notebook and the manuscript's Prior specification section). Any posterior
produced before this update used the earlier, non-elicited prior widths;
those results are stale and should be re-generated.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def build_model(coords, N0, ybar, se, prey_idx, sal_idx, a_sal_sigma=0.40, a_int_sigma=0.35):
    """Build the M1 functional-response model.

    Structural layer is a Holling type-II disc equation, ``mu = alpha * N0 /
    (1 + alpha * h * N0)``, with attack rate ``alpha`` (logit link, bounded
    in (0, 1)) and handling time ``h`` (log link, strictly positive). Both
    parameters carry the same ``{intercept, prey, salinity, prey x salinity
    interaction}`` structure, with main effects and the interaction
    parameterized as ZeroSumNormal (no reference level; each coefficient is
    a symmetric deviation from the grand mean). The measurement layer
    combines a known per-observation standard error (``se``) with an
    estimated process/lack-of-fit term (``tau``) in quadrature, feeding a
    Gamma likelihood on the observed cell mean.

    Parameters
    ----------
    coords : dict
        PyMC coordinate dictionary with keys ``"prey"``, ``"salinity"``,
        and ``"obs"``, as built from the factorized prey/salinity labels
        and the observation index.
    N0 : ndarray of float, shape (n_obs,)
        Offered prey density for each observation.
    ybar : ndarray of float, shape (n_obs,)
        Observed mean number of prey removed for each observation.
    se : ndarray of float, shape (n_obs,)
        Known standard error of ``ybar`` for each observation (from the
        four-replicate design).
    prey_idx : ndarray of int, shape (n_obs,)
        Zero-based prey index for each observation, aligned with
        ``coords["prey"]``.
    sal_idx : ndarray of int, shape (n_obs,)
        Zero-based salinity index for each observation, aligned with
        ``coords["salinity"]``.
    a_sal_sigma : float, default 0.40
        ZeroSumNormal scale for the salinity main effect on attack rate
        (logit scale). Default is the maximum-entropy value elicited in
        ``01_prior_elicitation.ipynb`` (R~3x salinity fold-change claim).
        Exposed as an argument, rather than hardcoded, specifically to
        support prior-sensitivity sweeps: power-scaling analysis flagged
        this parameter as showing prior-data conflict (see
        02_sampling_and_inference.ipynb), so its stability across a range
        of defensible widths is itself a diagnostic question, not just a
        single committed value.
    a_int_sigma : float, default 0.35
        ZeroSumNormal scale for the prey x salinity interaction on attack
        rate (logit scale). Default is the elicited value (half the
        prey main-effect scale, a background lean toward additivity).
        Exposed for the same prior-sensitivity-sweep reason as
        ``a_sal_sigma``: power-scaling analysis flagged every interaction
        cell as showing prior-data conflict.

    Returns
    -------
    pymc.Model
        The unfit model, ready for ``pm.sample_prior_predictive``,
        ``pm.sample``, or structural checks (``.debug()``,
        ``.to_graphviz()``).
    """
    with pm.Model(coords=coords) as fr_model:
        # Everything that could be swapped at predict-time lives in pm.Data
        N0_ = pm.Data("N0", N0, dims="obs")        # predictor AND upper bound
        ybar_ = pm.Data("ybar", ybar, dims="obs")   # observed cell mean
        se_ = pm.Data("se", se, dims="obs")         # known SE
        pix = pm.Data("prey_idx", prey_idx, dims="obs")
        six = pm.Data("sal_idx", sal_idx, dims="obs")

        # --- low-density capture fraction alpha, in (0,1) ------------------
        # BACKGROUND CONSTRAINT (not from this data): consumption cannot
        # exceed prey offered, so the low-density slope consumed/offered is
        # a fraction in (0,1). Hence the LOGIT scale; invlogit also forces
        # mu < N0 by construction.
        #
        # a0: with ZeroSumNormal effects (sum to 0), the intercept is the
        #     GRAND MEAN logit capture fraction across all cells.
        #     Normal(0, 1.5) => median fraction 0.5, central 95% ~ (0.05,
        #     0.95): an ignorance prior on a bounded fraction, symmetric,
        #     pre-data.
        a0 = pm.Normal("a0", mu=0, sigma=1.5)
        a_prey = pm.ZeroSumNormal("a_prey", sigma=0.70, dims="prey")
        a_sal = pm.ZeroSumNormal("a_sal", sigma=a_sal_sigma, dims="salinity")
        # interaction: deviation not explained by the two main effects,
        # sum-to-zero on BOTH margins (n_zerosum_axes=2). Has a sigma lower
        # than the main-effect sigma, encoding a background lean toward
        # additivity -- the interaction is a correction, expected to be the
        # smaller term until the data says otherwise.
        a_int = pm.ZeroSumNormal(
            "a_int", sigma=a_int_sigma, dims=("prey", "salinity"), n_zerosum_axes=2
        )
        alpha = pm.Deterministic(
            "\u03b1",
            pm.math.invlogit(a0 + a_prey[pix] + a_sal[six] + a_int[pix, six]),
            dims="obs",
        )

        # handling time h > 0 -> log scale. Same (prey, salinity,
        # interaction) structure as the attack rate, because the DAG routes
        # BOTH parameters through the same prey/predator osmotic states:
        # whatever lets salinity and prey identity (and their interaction)
        # move alpha, moves h too.
        # Priors are background-only, deliberately tighter than a's.
        # Rationale: (i) euryhaline species whose tested 10-30 g/L sits
        # inside its 5-40 tolerance, so handling shifts are expected to be
        # modest; (ii) the asymptote (1/h) is structurally weakly identified
        # by sub-saturating curves, so we lean on background and let
        # power-scaling sensitivity report which h-effects are
        # prior-driven rather than data-driven.
        h0 = pm.Normal("h0", np.log(1 / 40), 0.74)  # grand-mean log handling time
        h_prey = pm.ZeroSumNormal("h_prey", sigma=0.50, dims="prey")
        h_sal = pm.ZeroSumNormal("h_sal", sigma=0.30, dims="salinity")
        h_int = pm.ZeroSumNormal(
            "h_int", sigma=0.25, dims=("prey", "salinity"), n_zerosum_axes=2
        )
        h = pm.Deterministic(
            "h",
            pt.exp(h0 + h_prey[pix] + h_sal[six] + h_int[pix, six]),
            dims="obs",
        )
        pm.Deterministic("A", 1.0 / h, dims="obs")  # asymptote, per cell

        # --- Holling type II disc equation, native (attack rate, handling
        # time) form ---
        mu = pm.Deterministic(
            "\u03bc", alpha * N0_ / (1 + alpha * h * N0_), dims="obs"
        )
        tau = pm.Gamma("\u03c4", alpha=2.66, beta=0.99)
        sigma_eff = pm.Deterministic(
            "\u03c3_eff", pt.sqrt(tau**2 + se_**2), dims="obs"
        )

        pm.Gamma("likelihood", mu=mu, sigma=sigma_eff, observed=ybar_, dims="obs")

    return fr_model
