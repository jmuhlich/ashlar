import warnings
import attr
import numpy as np
import networkx as nx

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError:
    # FIXME Figure out what to do here.
    warnings.warn("plotting not available; please install matplotlib")
    pass


@attr.s(frozen=True)
class TileSetPlotter(object):
    """TileSet plotting helper

    Call one of the plot methods explicitly, or call the plotter itself for
    the default plot, `scatter`.
    """
    tileset = attr.ib()

    def __call__(self, **kwargs):
        self.scatter(**kwargs)

    def scatter(self, ax=None, **kwargs):
        """Create a scatter plot of the tile positions."""
        if ax is None:
            ax = plt.gca()
        y, x = self.tileset.positions.T
        ax.scatter(x, y, **kwargs)
        ax.set_aspect('equal')

    def rectangles(self, ax=None, **kwargs):
        """Draw a rectangle representing each tile's position and size."""
        for r in self.tileset.rectangles:
            draw_rectangle(r, ax, **kwargs)


@attr.s(frozen=True)
class RegistrationProcessPlotter(object):
    """RegistrationProcess plotting helper"""

    process = attr.ib()

    def __call__(self, **kwargs):
        self.graph(**kwargs)

    def graph(self, ax=None, **kwargs):
        """Draw the neighbors graph using the tile centers for layout."""
        defaults = dict(
            font_size=6, node_size=100, node_color='orange'
        )
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        g = self.process.graph
        pos = np.fliplr(self.process.tileset.centers)
        if ax is None:
            ax = plt.gca()
        nx.draw_networkx(g, ax=ax, pos=pos, with_labels=True, **kwargs)
        ax.set_aspect('equal')


@attr.s(frozen=True)
class RegistrationProcessExecutorPlotter(object):
    """RegistrationProcessExecutor plotting helper"""

    executor = attr.ib()

    def __call__(self, **kwargs):
        self.spanning_tree(**kwargs)

    def spanning_tree(self, ax=None, **kwargs):
        """Draw the spanning_tree graph using the tile centers for layout."""
        defaults = dict(
            font_size=6, node_size=100, node_color='orange'
        )
        for k, v in defaults.items():
            kwargs.setdefault(k, v)
        g = self.executor.spanning_tree_
        pos = np.fliplr(self.executor.process.tileset.centers)
        if ax is None:
            ax = plt.gca()
        nx.draw_networkx(g, ax=ax, pos=pos, with_labels=True, **kwargs)
        ax.set_aspect('equal')


def draw_rectangle(rect, ax=None, **kwargs):
    defaults = dict(color='black', fill=False, lw=0.5)
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    if ax is None:
        ax = plt.gca()
    xy = (rect.vector1.x, rect.vector1.y)
    w = rect.shape.x
    h = rect.shape.y
    mrect = mpatches.Rectangle(xy, w, h, **kwargs)
    ax.add_patch(mrect)
    ax.autoscale_view()
    ax.set_aspect('equal')
    return mrect


def draw_plane(plane, ax=None, **kwargs):
    defaults = dict(cmap='Greens')
    for k, v in defaults.items():
        kwargs.setdefault(k, v)
    v1 = plane.bounds.vector1
    v2 = plane.bounds.vector2
    extent = (v1.x, v2.x, v2.y, v1.y)
    if ax is None:
        ax = plt.gca()
    image = ax.imshow(plane.image, extent=extent, **kwargs)
    return image
