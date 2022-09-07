import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.patches as mpatches
import matplotlib.patheffects as mpatheffects
import numpy as np
import seaborn as sns


def generate_report(path, aligners):
    aligner0, *aligners = aligners
    plot_edge_scatter(aligner0)


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


def plot_edge_quality(
    aligner, img=None, show_tree=True, pos='metadata', im_kwargs=None, nx_kwargs=None
):
    if pos == 'metadata':
        centers = aligner.metadata.centers - aligner.metadata.origin
    elif pos == 'aligner':
        centers = aligner.centers
    else:
        raise ValueError("pos must be either 'metadata' or 'aligner'")
    if im_kwargs is None:
        im_kwargs = {}
    if nx_kwargs is None:
        nx_kwargs = {}
    final_nx_kwargs = dict(width=2, node_size=100, font_size=6)
    final_nx_kwargs.update(nx_kwargs)
    if show_tree:
        nrows, ncols = 1, 2
        if aligner.mosaic_shape[1] * 2 / aligner.mosaic_shape[0] > 2 * 4 / 3:
            nrows, ncols = ncols, nrows
    else:
        nrows, ncols = 1, 1
    fig = plt.figure()
    ax = plt.subplot(nrows, ncols, 1)
    draw_mosaic_image(ax, aligner, img, **im_kwargs)
    error = np.array([aligner._cache[tuple(sorted(e))][1]
                      for e in aligner.neighbors_graph.edges])
    # Manually center and scale data to 0-1, except infinity which is set to -1.
    # This lets us use the purple-green diverging color map to color the graph
    # edges and cause the "infinity" edges to disappear into the background
    # (which is itself purple).
    infs = error == np.inf
    error[infs] = -1
    if not infs.all():
        error_f = error[~infs]
        emin = np.min(error_f)
        emax = np.max(error_f)
        if emin == emax:
            # Always true when there's only one edge. Otherwise it's unlikely
            # but theoretically possible.
            erange = 1
        else:
            erange = emax - emin
        error[~infs] = (error_f - emin) / erange
    # Neighbor graph colored by edge alignment quality (brighter = better).
    nx.draw(
        aligner.neighbors_graph, ax=ax, with_labels=True,
        pos=np.fliplr(centers), edge_color=error, edge_vmin=-1, edge_vmax=1,
        edge_cmap=plt.get_cmap('PRGn'), **final_nx_kwargs
    )
    if show_tree:
        ax = plt.subplot(nrows, ncols, 2)
        draw_mosaic_image(ax, aligner, img, **im_kwargs)
        # Spanning tree with nodes at original tile positions.
        nx.draw(
            aligner.spanning_tree, ax=ax, with_labels=True,
            pos=np.fliplr(centers), edge_color='royalblue',
            **final_nx_kwargs
        )
    fig.set_facecolor('black')


def plot_edge_scatter(aligner, annotate=True):
    xdata = np.clip(aligner.all_errors, 0, 10)
    ydata = np.clip(
        [np.linalg.norm(v[0]) for v in aligner._cache.values()], 0.01, np.inf
    )
    pdata = np.clip(aligner.errors_negative_sampled, 0, 10)
    g = sns.JointGrid(x=xdata, y=ydata)
    g.plot_joint(sns.scatterplot, alpha=0.5, ax=g.ax_joint)
    _, xbins = np.histogram(np.hstack([xdata, pdata]), bins=40)
    sns.distplot(
        xdata, ax=g.ax_marg_x, kde=False, bins=xbins, norm_hist=True
    )
    sns.distplot(
        pdata, ax=g.ax_marg_x, kde=False, bins=xbins, norm_hist=True,
        hist_kws=dict(histtype='step')
    )
    g.ax_joint.axvline(aligner.max_error, c='k', ls=':')
    g.ax_joint.axhline(aligner.max_shift_pixels, c='k', ls=':')
    g.ax_joint.set_yscale('log')
    g.set_axis_labels('error', 'shift')
    if annotate:
        for pair, x, y in zip(aligner.neighbors_graph.edges, xdata, ydata):
            g.ax_joint.annotate(str(pair), (x, y), alpha=0.1)
    plt.tight_layout()


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
    draw_mosaic_image(ax, aligner, img, **im_kwargs)

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
    h, w = aligner.mosaic_shape
    ax.imshow(img, extent=(-0.5, w-0.5, h-0.5, -0.5), **kwargs)
