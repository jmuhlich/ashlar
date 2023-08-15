import copy
import fpdf
import functools
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import Image
import seaborn as sns
import skimage.exposure


cmap_yellow = mcolors.LinearSegmentedColormap(
    name="a_yellow",
    segmentdata={
        'red': ((0, 0.1, 0.1), (1, 1, 1)),
        'green': ((0, 0.1, 0.1), (1, 1, 1)),
        'blue': ((0, 0, 0), (1, 0.8, 0.8)),
    },
)


def generate_report(path, aligners):
    aligner0, *aligners = aligners
    pdf = PDF(unit="in", format="letter")
    pdf.set_font("Helvetica")

    pdf.add_page()
    fig = plot_edge_map(aligner0, aligner0.reader.thumbnail)
    fig.set_size_inches(7.5, 9)
    pdf.add_figure(fig, "Cycle 1: Tile pair alignment map")
    plt.close(fig)

    pdf.add_page()
    fig = plot_edge_quality(aligner0)
    fig.set_size_inches(7.5, 7.5)
    pdf.add_figure(fig, "Cycle 1: Tile pair alignment quality")
    plt.close(fig)

    for i, aligner in enumerate(aligners, 2):
        pdf.add_page()
        fig = plot_layer_map(aligner, aligner.reader.thumbnail)
        fig.set_size_inches(7.5, 9)
        pdf.add_figure(fig, f"Cycle {i}: Cycle alignment map")
        plt.close(fig)

    pdf.output(path)


class PDF(fpdf.FPDF):

    def add_figure_title(self, txt):
        with self.local_context(font_size=16, font_style="B"):
            self.cell(txt=txt, w=0, align="C", new_x="LEFT", new_y="NEXT")

    def add_figure(self, fig, title):
        self.add_figure_title(title)
        if fig.findobj(mimage.AxesImage):
            fig.set_dpi(300)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            img = Image.fromarray(np.asarray(canvas.buffer_rgba()))
        else:
            img = io.BytesIO()
            fig.savefig(img, format="svg")
        self.image(img, w=self.epw)


def plot_edge_map(
    aligner,
    img=None,
    pos="metadata",
    cmap=None,
    width=None,
    node_size=None,
    font_size=None,
    im_kwargs=None,
    nx_kwargs=None,
):
    """Plot neighbor graph colored by edge alignment quality"""

    if pos == "metadata":
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == "aligner":
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    cmap = cmap or cmap_yellow
    im_kwargs = im_kwargs or {}
    nx_kwargs = nx_kwargs or {}
    fig, ax = plt.subplots()
    draw_mosaic_image(ax, aligner, img, cmap=cmap, **im_kwargs)
    error = np.array([
        aligner._cache[tuple(sorted(e))][1]
        for e in aligner.neighbors_graph.edges
    ])
    efinite = error[error < np.inf]
    emin, emax = (efinite.min(), efinite.max()) if len(efinite) > 0 else (0, 0)
    in_tree = [
        e in aligner.spanning_tree.edges
        for e in aligner.neighbors_graph.edges
    ]
    diameter = nx.diameter(aligner.neighbors_graph)
    drange = [10, 60]
    interp = functools.partial(np.interp, diameter, [10, 60])
    width = width or interp([4, 1])
    node_size = node_size or interp([100, 8])
    font_size = font_size or interp([6, 2])
    width = np.where(in_tree, width, width * 0.75)
    style = np.where(in_tree, "solid", "dotted")
    edge_cmap = copy.copy(mcm.Blues)
    edge_cmap.set_over((0.1, 0.1, 0.1))
    g = aligner.neighbors_graph
    pos = np.fliplr(centers)
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        node_color="silver",
        node_size=node_size,
        edgecolors=None,
        **nx_kwargs,
    )
    ec = nx.draw_networkx_edges(
        g,
        pos,
        ax=ax,
        edge_color=error,
        edge_vmin=emin,
        edge_vmax=emax,
        edge_cmap=edge_cmap,
        width=width,
        style=style,
        **nx_kwargs,
    )
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=font_size, **nx_kwargs)
    draw_borders(ax, aligner, pos)
    cbar = fig.colorbar(
        mcm.ScalarMappable(mcolors.Normalize(emin, emax), edge_cmap),
        extend="max",
        label="Error (-log NCC)",
        location="right",
        shrink=0.5,
        ax=ax,
    )
    ax.set_frame_on(False)
    ax.margins(0)
    fig.tight_layout()
    return fig


def plot_edge_shifts(aligner, img=None, bounds=True, im_kwargs=None):
    if im_kwargs is None:
        im_kwargs = {}
    fig = plt.figure()
    ax = plt.gca()
    draw_mosaic_image(ax, aligner, img, **im_kwargs)
    h, w = aligner.reader.metadata.size
    if bounds:
        # Bounding boxes denoting new tile positions.
        for xy in np.fliplr(aligner.positions):
            rect = mpatches.Rectangle(xy, w, h, color='black', fill=False,
                                      lw=0.5)
            ax.add_patch(rect)
    # Compute per-edge relative shifts from tile positions.
    edges = np.array(list(aligner.spanning_tree.edges))
    dist = aligner.metadata.positions - aligner.positions
    shifts = dist[edges[:, 0]] - dist[edges[:, 1]]
    shift_distances = np.linalg.norm(shifts, axis=1)
    # Spanning tree with nodes at new tile positions, edges colored by shift
    # distance (brighter = farther).
    nx.draw(
        aligner.spanning_tree, ax=ax, with_labels=True,
        pos=np.fliplr(aligner.centers), edge_color=shift_distances,
        edge_cmap=plt.get_cmap('Blues_r'), width=2, node_size=100, font_size=6
    )
    fig.set_facecolor('black')


def plot_edge_quality(aligner, annotate=True):
    xdata = np.clip(aligner.all_errors, 0, 10)
    ydata = (
        np.linalg.norm([v[0] for v in aligner._cache.values()], axis=1)
        .clip(0.01, np.inf)
        * aligner.metadata.pixel_size
    )
    pdata = np.clip(aligner.errors_negative_sampled, 0, 10)
    g = sns.JointGrid(x=xdata, y=ydata)
    g.plot_joint(sns.scatterplot, alpha=0.5, ec="none", ax=g.ax_joint)
    histplot = functools.partial(
        sns.histplot,
        bins=40,
        stat="density",
        kde=True,
        color="tab:blue",
        ec="white",
    )
    histplot(x=xdata, ax=g.ax_marg_x)
    histplot(y=ydata, ax=g.ax_marg_y, log_scale=True)
    sns.kdeplot(x=pdata, ax=g.ax_marg_x, color="salmon")
    g.ax_joint.axvline(aligner.max_error, c="k", ls=":")
    g.ax_marg_x.axvline(aligner.max_error, c="k", ls=":")
    g.ax_joint.axhline(aligner.max_shift, c="k", ls=":")
    g.ax_marg_y.axhline(aligner.max_shift, c="k", ls=":")
    g.set_axis_labels("Error (-log NCC)", "Shift distance (\u00B5m)")
    if annotate:
        for pair, x, y in zip(aligner.neighbors_graph.edges, xdata, ydata):
            g.ax_joint.annotate(str(pair), (x, y), alpha=0.25, size=6)
    g.figure.tight_layout()
    return g.figure


def plot_layer_shifts(aligner, img=None, im_kwargs=None):
    if im_kwargs is None:
        im_kwargs = {}
    fig = plt.figure()
    ax = plt.gca()
    draw_mosaic_image(ax, aligner, img, **im_kwargs)
    h, w = aligner.metadata.size
    # Bounding boxes denoting new tile positions.
    for xy in np.fliplr(aligner.positions):
        rect = mpatches.Rectangle(xy, w, h, color='black', fill=False, lw=0.5)
        ax.add_patch(rect)
    # Neighbor graph with edges hidden, i.e. just show nodes.
    nx.draw(
        aligner.neighbors_graph, ax=ax, with_labels=True,
        pos=np.fliplr(aligner.centers), edge_color='none',
        node_size=100, font_size=6
    )
    fig.set_facecolor('black')


def plot_layer_map(
    aligner,
    img=None,
    pos="metadata",
    cmap=None,
    width=None,
    node_size=None,
    font_size=None,
    im_kwargs=None,
    nx_kwargs=None,
):
    """Plot tile shift distance and direction"""

    if pos == "metadata":
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == "aligner":
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    cmap = cmap or cmap_yellow
    im_kwargs = im_kwargs or {}
    nx_kwargs = nx_kwargs or {}
    fig, ax = plt.subplots()
    draw_mosaic_image(ax, aligner, img, cmap=cmap, **im_kwargs)

    pixel_size = aligner.reader.metadata.pixel_size
    shifts = np.linalg.norm(aligner.shifts, axis=1) * pixel_size
    skeep = shifts[~aligner.discard]
    smin, smax = (skeep.min(), skeep.max()) if len(skeep) > 0 else (0, 0)
    # Map discards to the "over" color in the cmap.
    shifts[aligner.discard] = smax + 1
    # Reorder to match the graph's internal node ordering.
    node_values = shifts[np.array(aligner.neighbors_graph)]

    diameter = nx.diameter(aligner.neighbors_graph)
    drange = [10, 60]
    interp = functools.partial(np.interp, diameter, drange)
    node_size = node_size or interp([100, 8])
    font_size = font_size or interp([6, 2])
    node_cmap = mcm.Greens.with_extremes(over="#252525")
    g = aligner.neighbors_graph
    pos = np.fliplr(centers)
    qlen = np.min(aligner.metadata.size) * 0.45
    q_angles = np.rad2deg(np.arctan2(*(aligner.shifts * [-1, 1]).T))
    reference_offset = (
        aligner.cycle_offset
        + aligner.metadata.origin
        - aligner.reference_aligner.metadata.origin
    )
    reference_corners = (
        aligner.reference_aligner.metadata.centers
        - aligner.reference_aligner.metadata.origin
        + aligner.reference_aligner.metadata.size / 2
    )
    reference_size = np.max(reference_corners, axis=0)

    ax.add_patch(
        mpatches.Rectangle(
            -reference_offset[::-1],
            *reference_size[::-1],
            color=mcm.Blues(0.6),
            lw=2,
            linestyle="--",
            fill=False,
        )
    )
    ax.quiver(
        pos[:, 0],
        pos[:, 1],
        [qlen] * len(pos),
        [0] * len(pos),
        shifts,
        cmap=node_cmap,
        clim=(smin, smax),
        angles=q_angles,
        scale_units="x",
        scale=1,
        headwidth=1,
        headlength=1,
        headaxislength=1,
    )
    nx.draw_networkx_nodes(
        g,
        pos,
        ax=ax,
        cmap=node_cmap,
        vmin=smin,
        vmax=smax,
        node_color=node_values,
        node_size=node_size,
        edgecolors=None,
        **nx_kwargs,
    )
    draw_labels = functools.partial(
        nx.draw_networkx_labels,
        pos=pos,
        ax=ax,
        font_size=font_size,
        **nx_kwargs,
    )
    draw_labels(g.subgraph(np.nonzero(aligner.discard)[0]), font_color="gray")
    draw_labels(g.subgraph(np.nonzero(~aligner.discard)[0]), font_color="k")
    draw_borders(ax, aligner, pos)
    cbar = fig.colorbar(
        mcm.ScalarMappable(mcolors.Normalize(smin, smax), node_cmap),
        extend="max",
        label="Shift (\xb5m)",
        location="right",
        shrink=0.5,
        ax=ax,
    )
    ax.set_frame_on(False)
    ax.margins(0)
    fig.tight_layout()
    return fig


def draw_mosaic_image(ax, aligner, img, **kwargs):
    if img is None:
        img = [[0]]
    cmax = np.max(aligner.metadata.centers - aligner.metadata.origin, axis=0)
    h, w = cmax + aligner.metadata.size / 2
    if "vmin" not in kwargs:
        kwargs["vmin"] = np.percentile(img, 1)
    if "vmax" not in kwargs:
        kwargs["vmax"] = np.percentile(img, 99)
    ax.imshow(img, extent=(-0.5, w-0.5, h-0.5, -0.5), **kwargs)


def draw_borders(ax, aligner, pos):
    rh, rw = aligner.metadata.size
    for x, y in pos - (rw / 2, rh / 2):
        rect = mpatches.Rectangle(
            (x, y),
            rw,
            rh,
            color=(0.2, 0.2, 0.2),
            fill=False,
            lw=0.25,
            zorder=0.5,
        )
        ax.add_patch(rect)
