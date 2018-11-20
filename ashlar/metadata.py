from __future__ import division, print_function
from abc import abstractmethod
import attr
import numpy as np
import scipy.spatial.distance
import networkx as nx
from . import util, geometry, plot


@attr.s(frozen=True)
class ImageReader(util.ABC):
    dtype = attr.ib()
    pixel_size = attr.ib()
    num_channels = attr.ib()

    @abstractmethod
    def read(self, image_number, channel):
        """Return one image plane from a multi-channel image series."""
        pass


@attr.s(frozen=True)
class TileSet(object):
    """Physical layout of a set of image tiles and access to the pixels.

    Tile positions and shapes are always in microns, not pixels!

    """
    tile_shape = attr.ib(converter=util.array_copy_immutable)
    positions = attr.ib(converter=util.array_copy_immutable)
    _reader = attr.ib(validator=attr.validators.instance_of(ImageReader))

    @property
    def grid_shape(self):
        """Return shape of tile grid, if tile positions do form a grid."""
        pos = self.positions
        shape = np.array([len(set(pos[:, d])) for d in range(2)])
        if np.prod(shape) != len(self):
            raise ValueError("Series positions do not form a grid")
        return shape

    @property
    def centers(self):
        """Return array of Y, X tile centers."""
        return self.positions + self.tile_shape / 2

    @property
    def origin(self):
        """Return array of minimum Y, X coordinates."""
        return geometry.Vector.from_ndarray(np.min(self.positions, axis=0))

    @property
    def rectangles(self):
        """Return list of Rectangles representing tiles."""
        ts = geometry.Vector.from_ndarray(self.tile_shape)
        rectangles = [
            geometry.Rectangle.from_shape(geometry.Vector.from_ndarray(p), ts)
            for p in self.positions
        ]
        return rectangles

    @property
    def plot(self):
        """Return plotter utility object (see plot.TileSetPlotter)."""
        return plot.TileSetPlotter(self)

    def build_neighbors_graph(self, cutoff=50, bias=0):
        """Return graph of neighboring tiles.

        Tiles are considered neighboring if the overlap area of their bounding
        rectangles is greater than the `cutoff` percentile of all overlapping
        tiles. The `bias` parameter will expand or contract the rectangles for a
        more or less inclusive test.

        The default `cutoff` percentile of 50 was chosen to reject diagonally
        adjacent tiles in a regular grid. As there are slightly more up-down and
        left-right neighbors than diagonal neighbors in a grid, the 50th
        percentile will correspond to an up-down or left-right neighbor
        intersection area. Very unusual tile position collections may require
        tuning of this parameter.

        The default `bias` value of 0 will only consider strictly overlapping
        tiles. Increasing this parameter will also consider touching or even
        disjoint tiles. The typical use case for increasing `bias` is for data
        sets where neighboring stage positions just touch, but due to stage
        position error the imaged regions do have some actual overlap that can
        be registered. Specifying a small bias value will include these touching
        neighbors in the neighbors graph.

        """
        recs = [r.inflate(bias) for r in self.rectangles]
        overlaps = [[r1.intersection(r2).area for r2 in recs] for r1 in recs]
        mask = np.tri(len(self), k=-1)
        overlaps = np.where(mask, overlaps, 0)
        idxs_nonzero = np.nonzero(overlaps)
        if len(idxs_nonzero[0]) > 0:
            cutoff_value = np.percentile(overlaps[idxs_nonzero], cutoff)
        else:
            # Should we raise an exception in this case?
            cutoff_value = np.inf
        idxs = np.nonzero(overlaps >= cutoff_value)
        graph = nx.from_edgelist(zip(*idxs))
        return graph

    def get_tile(self, tile_number, channel):
        image = self._reader.read(tile_number, channel)
        bounds = self.rectangles[tile_number]
        tile = Tile(image, bounds, self._reader.pixel_size)
        return tile

    def __len__(self):
        return len(self.positions)


@attr.s(frozen=True)
class Tile(object):
    image = attr.ib(validator=attr.validators.instance_of(np.ndarray))
    bounds = attr.ib(validator=attr.validators.instance_of(geometry.Rectangle))
    pixel_size = attr.ib()

    def intersection(self, other, min_overlap=0):
        bounds = self.bounds.intersection(other.bounds)
        shape = bounds.shape
        min_width = min(shape.y, shape.x)
        padding = min_overlap - min_width
        if padding > 0:
            bounds = self.bounds.intersection(bounds.inflate(padding))
        crop_region = (bounds - self.bounds.vector1) / self.pixel_size
        image = self.image[crop_region.as_slice]
        return Tile(image, bounds, self.pixel_size)
