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
    rh, rw = aligner.metadata.size
    for x, y in pos:
        x -= rw / 2
        y -= rh / 2
        rect = mpatches.Rectangle(
            (x, y), rw, rh, color='silver', alpha=0.25, fill=False, lw=0.25, zorder=0.5
        )
        ax.add_patch(rect)
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


def plot_layer_quality(
    aligner, img=None, scale=1.0, artist='patches', annotate=True, im_kwargs=None
):
    if im_kwargs is None:
        im_kwargs = {}
    fig = plt.figure()
    ax = plt.gca()
    draw_mosaic_image(ax, aligner, img, cmap=cmap_yellow, **im_kwargs)

    h, w = aligner.metadata.size
    positions, centers, shifts = aligner.positions, aligner.centers, aligner.shifts

    if scale != 1.0:
        h, w, positions, centers, shifts = [
            scale * i for i in [h, w, positions, centers, shifts]
        ]

    # Bounding boxes denoting new tile positions.
    color_index = skimage.exposure.rescale_intensity(
        aligner.errors, out_range=np.uint8
    ).astype(np.uint8)
    color_map = mcm.magma_r
    for xy, c_idx in zip(np.fliplr(positions), color_index):
        rect = mpatches.Rectangle(
            xy, w, h, color=color_map(c_idx), fill=False, lw=0.5
        )
        ax.add_patch(rect)
    
    # Annotate tile numbering.
    if annotate:
        for idx, (x, y) in enumerate(np.fliplr(positions)):
            text = plt.annotate(str(idx), (x+0.1*w, y+0.9*h), alpha=0.7)
            # Add outline to text for better contrast in different background color.
            text_outline = mpatheffects.Stroke(linewidth=1, foreground='#AAA')
            text.set_path_effects(
                [text_outline, mpatheffects.Normal()]
            )

    if artist == 'quiver':
        ax.quiver(
            *centers.T[::-1], *shifts.T[::-1], aligner.discard,
            units='dots', width=2, scale=1, scale_units='xy', angles='xy',
            cmap='Greys'
        )
    if artist == 'patches':
        for xy, dxy, is_discarded in zip(
            np.fliplr(centers), np.fliplr(shifts), aligner.discard
        ):
            arrow = mpatches.FancyArrowPatch(
                xy, np.array(xy) + np.array(dxy), 
                arrowstyle='->', color='0' if is_discarded else '1',
                mutation_scale=8,
                )
            ax.add_patch(arrow)
    ax.axis('off')


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
