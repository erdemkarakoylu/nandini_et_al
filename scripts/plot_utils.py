"""Visualization helpers for the *Cordylophora* functional-response project.

Contains the two-panel causal/statistical DAG builder (McElreath dashed/
solid convention: dashed = unobserved or latent, solid = observed or
experimentally set) and a thin wrapper around ``arviz_plots.plot_forest``
for rendering named posterior contrasts as forest plots.
"""
import graphviz
import matplotlib.pyplot as plt
import arviz_plots as azp

# McElreath convention: DASHED = unobserved / latent ; SOLID = observed / data.
# Light fills encode ROLE; the border style carries observed-vs-latent.
DESIGN = dict(shape="box", style="filled,solid", fillcolor="#dbeafe", color="#1e3a8a")
OUTCOME = dict(shape="box", style="filled,solid", fillcolor="#dcfce7", color="#166534")
PARAM = dict(shape="ellipse", style="filled,dashed", fillcolor="#fef3c7", color="#92400e")
LATENT = dict(shape="ellipse", style="dashed", color="#6b7280")


def _node(g, name, label, kind):
    """Add a styled node to a graphviz graph.

    Parameters
    ----------
    g : graphviz.Digraph or graphviz.Digraph subgraph context
        Graph (or subgraph) to add the node to.
    name : str
        Node identifier.
    label : str
        Text displayed on the node.
    kind : dict
        Style keyword arguments (shape, fill, color, etc.), typically one
        of the module-level ``DESIGN``, ``OUTCOME``, ``PARAM``, or
        ``LATENT`` dicts.
    """
    g.node(name, label, **kind)


def two_panel_dag(variant="M1", save_name="fig_m1"):
    """Build the two-panel causal + statistical DAG for the functional-
    response model.

    The left panel ("what we believe") is the causal skeleton shared by
    every model variant: salinity and prey type are experimenter-set design
    variables acting on the latent attack rate and handling time, salinity
    routed only through unobserved prey- and predator-side osmotic states.
    The right panel ("what we fit") shows the measurement layer, which is
    the only part that changes across variants.

    Parameters
    ----------
    variant : {"M0", "M1", "M2"}, default "M1"
        Which measurement-layer variant to draw in the right panel.
        ``"M0"``: known-SE-only Gamma likelihood (structural baseline).
        ``"M1"``: two-component (process + measurement) Gamma likelihood,
        the model currently fit. ``"M2"``: bounded-support Beta likelihood
        on the proportion consumed (planned, not yet implemented).
    save_name : str, default "fig_m1"
        Base filename used if the caller subsequently calls ``.render()``
        on the returned graph. Not used internally by this function.

    Returns
    -------
    graphviz.Digraph
        The assembled graph. Renders inline in Jupyter; call
        ``.render(path, format=...)`` to save to disk.
    """
    assert variant in {"M0", "M1", "M2"}
    g = graphviz.Digraph(f"fr_dag_{variant}", format="png")
    g.attr(
        rankdir="LR", fontname="Helvetica", labelloc="t", fontsize="16",
        label="Cordylophora functional response",
    )
    g.attr("edge", color="#555555", arrowsize="0.7")

    # ---- Panel 1: causal ("what we believe") ----
    with g.subgraph(name="cluster_causal") as c:
        c.attr(
            label="what we believe (causal)", style="rounded",
            color="#9ca3af", fontcolor="#374151", fontsize="13",
        )
        _node(c, "S", "Salinity\n(set)", DESIGN)
        _node(c, "P", "Prey type\n(set)", DESIGN)
        _node(c, "Uprey", "prey osmotic\nstate", LATENT)
        _node(c, "Upred", "predator osmotic\nstate", LATENT)
        c.edge("S", "Uprey")
        c.edge("S", "Upred")
        c.edge("P", "Uprey")

    # ---- bridge: latent Holling parameters ----
    _node(g, "alpha", "\u03b1  attack rate", PARAM)
    _node(g, "h", "h  handling time", PARAM)
    g.edge("P", "alpha")
    g.edge("P", "h")  # direct morphology effect
    g.edge("Uprey", "alpha")
    g.edge("Uprey", "h")
    g.edge("Upred", "alpha")
    g.edge("Upred", "h")

    # ---- Panel 2: model ("what we fit") ----
    with g.subgraph(name="cluster_model") as m:
        m.attr(
            label="what we fit (statistical)", style="rounded",
            color="#9ca3af", fontcolor="#374151", fontsize="13",
        )
        _node(m, "N0", "N0 density\n(set)", DESIGN)
        _node(m, "mu", "\u03bc = \u03b1N0/(1+\u03b1hN0)", LATENT)
        if variant == "M0":
            _node(m, "se", "se (known)", DESIGN)
            _node(m, "y", "\u0233  observed", OUTCOME)
            m.edge("mu", "y")
            m.edge("se", "y")
        elif variant == "M1":
            _node(m, "se", "se (known)", DESIGN)
            _node(m, "tau", "\u03c4  process SD", LATENT)
            _node(m, "sigma", "\u03c3_eff = \u221a(\u03c4\u00b2+se\u00b2)", LATENT)
            _node(m, "y", "\u0233  observed", OUTCOME)
            m.edge("tau", "sigma")
            m.edge("se", "sigma")
            m.edge("mu", "y")
            m.edge("sigma", "y")
        else:  # M2
            _node(m, "p", "p = \u03bc/N0\n(fraction)", LATENT)
            _node(m, "phi", "\u03c6  dispersion", LATENT)
            _node(m, "y", "\u0233/N0  observed\n(proportion)", OUTCOME)
            m.edge("p", "y")
            m.edge("phi", "y")

    g.edge("alpha", "mu")
    g.edge("h", "mu")
    g.edge("N0", "mu")
    if variant == "M2":
        g.edge("mu", "p")
        g.edge("N0", "p")

    return g


def render_forest(dataset, var_name, title, ref_line=0.0, figsize=None,
                   save_path=None, save_kwargs=None):
    """Render a labeled forest plot of posterior contrasts with a reference
    line.

    Thin wrapper around ``arviz_plots.plot_forest`` that applies the row
    labeling (from the dataset's ``contrast`` coordinate rather than the
    variable name), adds a titled reference line at the "no effect" value,
    and titles the figure.

    Parameters
    ----------
    dataset : xarray.Dataset
        Dataset with a single data variable named ``var_name``, carrying
        ``(chain, draw, contrast)`` dims, where ``contrast`` has string
        coordinates naming each row (see
        :func:`stats_utils.named_contrast_dataset`).
    var_name : str
        Name of the data variable in ``dataset`` to plot.
    title : str
        Figure title.
    ref_line : float, default 0.0
        Value at which to draw a vertical dashed reference line — 0 for a
        difference contrast, 1 for a ratio contrast.
    figsize : (float, float), optional
        Figure size in inches, passed through to the underlying
        ``plt.subplots()`` call via ``figure_kwargs``.
    save_path : str or pathlib.Path, optional
        If given, save the figure here via ``PlotCollection.savefig``.
    save_kwargs : dict, optional
        Extra keyword arguments for ``savefig`` (e.g. ``dpi``, ``format``,
        ``bbox_inches``). Ignored if `save_path` is not given.

    Returns
    -------
    arviz_plots.plot_collection.PlotCollection
        The plot collection returned by ``plot_forest``.

    Notes
    -----
    Figure size, legends, and saving are three separate concerns in
    ``arviz_plots`` 1.x and are not interchangeable keyword arguments:
    size goes through ``figure_kwargs`` at plot-call time, legends are
    added after the fact via ``PlotCollection.add_legend(dim, ...)``, and
    saving is a further separate ``PlotCollection.savefig(path, ...)``
    call forwarding to matplotlib. ``plot_forest`` has no legend by
    default (a single contrast series), so no ``add_legend`` call is made
    here; see :func:`render_prior_posterior` for a plot that needs one.
    """
    figure_kwargs = {"figsize": figsize} if figsize is not None else {}
    pc = azp.plot_forest(
        dataset, var_names=[var_name], combined=True, labels=["contrast"],
        figure_kwargs=figure_kwargs,
    )
    plt.axvline(ref_line, color="r", ls="--")
    pc.add_title(title)
    pc.viz["figure"].item().tight_layout()
    if save_path is not None:
        pc.savefig(save_path, **(save_kwargs or {}))
    return pc


def render_prior_posterior(dataset, var_names, title=None, figsize=None,
                            legend_loc="best", save_path=None, save_kwargs=None):
    """Render a prior-vs-posterior overlay with a legend, sized and saved
    consistently with :func:`render_forest`.

    Thin wrapper around ``arviz_plots.plot_prior_posterior``.

    Parameters
    ----------
    dataset : xarray.DataTree or arviz.InferenceData
        Object with both ``prior`` and ``posterior`` groups for the
        requested variables.
    var_names : list of str
        Variables to plot.
    title : str, optional
        Figure title. If not given, no title is added.
    figsize : (float, float), optional
        Figure size in inches.
    legend_loc : str, default "best"
        Passed to the legend as ``loc`` (matplotlib legend location string).
        The color aesthetic distinguishing prior from posterior is mapped
        to a dimension literally named ``"group"``.
    save_path : str or pathlib.Path, optional
        If given, save the figure here via ``PlotCollection.savefig``.
    save_kwargs : dict, optional
        Extra keyword arguments for ``savefig``.

    Returns
    -------
    arviz_plots.plot_collection.PlotCollection
        The plot collection returned by ``plot_prior_posterior``.

    Notes
    -----
    ``plot_prior_posterior`` adds its own "group" (prior vs. posterior)
    legend internally by default, attached to the *figure* (not to any
    individual axes). The documented way to configure it,
    ``visuals={"legend": {...}}``, is broken in the installed
    ``arviz_plots`` (1.2.0): it validates `visuals` against the schema of
    the internal ``plot_dist`` call, which does not accept a ``"legend"``
    key, before ever reaching the code that would handle it — so passing
    it always raises. This wrapper instead lets the default legend be
    created, then repositions the existing ``matplotlib.legend.Legend``
    object directly via ``Legend.set_loc``, rather than adding a second,
    independently-placed legend on top of it.
    """
    figure_kwargs = {"figsize": figsize} if figsize is not None else {}
    pc = azp.plot_prior_posterior(
        dataset, var_names=var_names, figure_kwargs=figure_kwargs
    )
    fig = pc.viz["figure"].item()
    if fig.legends:
        fig.legends[0].set_loc(legend_loc)
    if title is not None:
        pc.add_title(title)
    fig.tight_layout()
    if save_path is not None:
        pc.savefig(save_path, **(save_kwargs or {}))
    return pc
