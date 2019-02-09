from __future__ import division, print_function
from abc import abstractmethod
import attr
import attr.validators as av
import numpy as np
import scipy.spatial.distance
import networkx as nx
from . import util, geometry, plot
from .util import attrib, cached_property


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
    """Physical layout of a list of image tiles and access to their pixels.

    Tile positions and shapes are always in microns, not pixels!

    """
    tile_shape = attrib(
        converter=util.array_copy_immutable,
        doc="Array of shape (2,) with the Y, X tile dimensions in microns."
    )
    positions = attrib(
        converter=util.array_copy_immutable,
        doc="Array of shape (N, 2) with the Y, X tile positions in microns."
    )
    _reader = attrib(
        validator=attr.validators.instance_of(ImageReader),
        doc="ImageReader instance for pixel data access."
    )

    @cached_property
    def grid_shape(self):
        """Shape of tile grid, if tile positions do form a grid."""
        pos = self.positions
        shape = np.array([len(set(pos[:, d])) for d in range(2)])
        if np.prod(shape) != len(self):
            raise ValueError("Series positions do not form a grid")
        return shape

    @cached_property
    def centers(self):
        """Array of Y, X tile centers."""
        return self.positions + self.tile_shape / 2

    @cached_property
    def origin(self):
        """Array of minimum Y, X coordinates."""
        return geometry.Vector.from_ndarray(np.min(self.positions, axis=0))

    @cached_property
    def rectangles(self):
        """List of Rectangles representing tiles."""
        ts = geometry.Vector.from_ndarray(self.tile_shape)
        rectangles = [
            geometry.Rectangle.from_shape(geometry.Vector.from_ndarray(p), ts)
            for p in self.positions
        ]
        return rectangles

    @cached_property
    def plot(self):
        """Plotter utility object (see plot.TileSetPlotter)."""
        return plot.TileSetPlotter(self)

    def build_neighbors_graph(self, cutoff, bias):
        """Return graph of neighboring tiles.

        Tiles are considered neighboring if the overlap area of their bounding
        rectangles is greater than the `cutoff` percentile of all overlapping
        tiles. The `bias` parameter will expand or contract the rectangles for a
        more or less inclusive test.

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
        """Return Tile object for a given tile number and channel."""
        image = self._reader.read(tile_number, channel)
        bounds = self.rectangles[tile_number]
        plane = Plane(image, bounds, self._reader.pixel_size)
        tile = Tile(plane, tile_number, channel)
        return tile

    def __len__(self):
        return len(self.positions)


@attr.s(frozen=True)
class Plane(object):
    """A raster image and its physical dimensions and location."""

    image = attrib(
        validator=attr.validators.instance_of(np.ndarray),
        doc="Numpy array containing the image pixels."
    )
    bounds = attrib(
        validator=attr.validators.instance_of(geometry.Rectangle),
        doc="Rectangle representing the physical dimensions of the image."
    )
    pixel_size = attrib(converter=float, doc="Pixel size in microns.")

    def intersection(self, other, min_overlap=0):
        """Return the intersection of two Planes as another Plane."""
        if self.pixel_size != other.pixel_size:
            raise ValueError("Planes have different pixel sizes")
        bounds = self.bounds.intersection(other.bounds)
        shape = bounds.shape
        min_width = min(shape.y, shape.x)
        padding = min_overlap - min_width
        if padding > 0:
            bounds = self.bounds.intersection(bounds.inflate(padding))
        crop_region = (bounds - self.bounds.vector1) / self.pixel_size
        image = self.image[crop_region.as_slice]
        return Plane(image, bounds, self.pixel_size)


@attr.s(frozen=True)
class Tile(object):
    """A Plane taken from a specific series and channel in a collection."""

    plane = attrib(
        validator=av.instance_of(Plane),
        doc="Plane holding the image and physical position."
    )
    index = attrib(
        validator=av.instance_of(int),
        doc="Index of the tile within its collection."
    )
    channel = attrib(
        validator=av.instance_of(int),
        doc="Index of the image channel this tile represents."
    )
