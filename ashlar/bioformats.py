from __future__ import division, print_function
import warnings
import threading
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib
import numpy as np
import attr
from . import metadata
from .util import cached_property
import time

import jnius_config
if not jnius_config.vm_running:
    pkg_root = pathlib.Path(__file__).parent.resolve()
    bf_jar_path = pkg_root / 'jars' / 'loci_tools.jar'
    if not bf_jar_path.exists():
        raise RuntimeError("loci_tools.jar missing from distribution"
                           " (expected it at %s)" % bf_jar_path)
    jnius_config.add_classpath(str(bf_jar_path))
    # Prevent Kryo serialization library from triggering warnings on Java 9+.
    jnius_config.add_options("--illegal-access=deny")
import jnius


File = jnius.autoclass('java.io.File')
DebugTools = jnius.autoclass('loci.common.DebugTools')
IFormatReader = jnius.autoclass('loci.formats.IFormatReader')
Memoizer = jnius.autoclass('loci.formats.Memoizer')
MetadataRetrieve = jnius.autoclass('ome.xml.meta.MetadataRetrieve')
ServiceFactory = jnius.autoclass('loci.common.services.ServiceFactory')
OMEXMLService = jnius.autoclass('loci.formats.services.OMEXMLService')
ChannelSeparator = jnius.autoclass('loci.formats.ChannelSeparator')
UNITS = jnius.autoclass('ome.units.UNITS')
DebugTools.enableLogging("ERROR")

pixel_dtypes = {
    'uint8': np.dtype(np.uint8),
    'uint16': np.dtype(np.uint16),
}
ome_dtypes = {v: k for k, v in pixel_dtypes.items()}


@attr.s(frozen=True)
class BioformatsReader(object):
    path = attr.ib()
    bf_reader = attr.ib()
    bf_metadata = attr.ib()
    cache_directory = attr.ib(default=None)
    _thread_local = attr.ib(init=False, factory=threading.local)
    _lock = attr.ib(init=False, factory=threading.Lock)

    @classmethod
    def from_path(cls, path, cache_directory=None):
        """Return a new BioformatsReader given a path to a file."""
        factory = ServiceFactory()
        service = jnius.cast(OMEXMLService, factory.getInstance(OMEXMLService))
        bf_metadata = service.createOMEXMLMetadata()
        bf_reader = cls.get_bf_reader(cache_directory)
        bf_reader.setMetadataStore(bf_metadata)
        bf_reader.setId(path)
        bf_metadata = jnius.cast(MetadataRetrieve, bf_metadata)
        return cls(path, bf_reader, bf_metadata, cache_directory)

    @classmethod
    def get_bf_reader(cls, cache_directory):
        """Return a new ChannelSeparator, with optional caching."""
        if cache_directory is None:
            bf_reader = ChannelSeparator()
        else:
            memo_dir = File(cache_directory)
            bf_reader = Memoizer(ChannelSeparator(), 0, memo_dir)
        return bf_reader

    @property
    def local_bf_reader(self):
        """Return thread-local clone of bf_reader.

        Setting self.cache_directory can dramatically speed up the
        initialization of the thread-local readers.

        """
        try:
            bf_reader = self._thread_local.bf_reader
        except AttributeError:
            bf_reader = self.get_bf_reader(self.cache_directory)
            self._thread_local.bf_reader = bf_reader
            bf_reader.setId(self.path)
        return bf_reader

    @cached_property
    def tileset(self):
        """Return a TileSet object representing this dataset."""
        return metadata.TileSet(
            self.tile_shape_microns, self.positions, self.image_reader
        )

    @cached_property
    def image_reader(self):
        """Return an ImageReader object for this dataset."""
        series_indices = range(self.num_tiles)
        return BioformatsImageReader(
            self.pixel_dtype, self.pixel_size, self.num_channels, self,
            series_indices
        )

    @cached_property
    def pixel_dtype(self):
        # FIXME verify all images have the same dtype.
        ome_dtype = self.bf_metadata.getPixelsType(0).value
        dtype = pixel_dtypes.get(ome_dtype)
        if dtype is None:
            raise ValueError("can't handle pixel type: '{}'".format(ome_dtype))
        return dtype

    @cached_property
    def pixel_size(self):
        # FIXME verify all images have the same pixel size.
        quantities = [
            self.bf_metadata.getPixelsPhysicalSizeY(0),
            self.bf_metadata.getPixelsPhysicalSizeX(0)
        ]
        values = [
            length_as_microns(q, "pixel size") for q in quantities
        ]
        if values[0] != values[1]:
            raise ValueError(
                "can't handle non-square pixels: ({}, {})".format(values)
            )
        return values[0]

    @cached_property
    def num_channels(self):
        # FIXME verify all images have the same number of channels.
        return self.bf_metadata.getChannelCount(0)

    @cached_property
    def tile_shape(self):
        # FIXME verify all images have the same shape.
        quantities = [
            self.bf_metadata.getPixelsSizeY(0),
            self.bf_metadata.getPixelsSizeX(0)
        ]
        shape = np.array([q.value for q in quantities], dtype=int)
        return shape

    @cached_property
    def tile_shape_microns(self):
        return self.tile_shape * self.pixel_size

    @cached_property
    def positions(self):
        positions = np.array([
            self.get_position(i) for i in range(self.num_tiles)
        ])
        return positions

    def get_position(self, idx):
        """Return stage position Y, X in microns of one image."""
        # FIXME verify all planes have the same X,Y position.
        quantities = [
            self.bf_metadata.getPlanePositionY(idx, 0),
            self.bf_metadata.getPlanePositionX(idx, 0)
        ]
        values = [
            length_as_microns(q, "stage coordinates") for q in quantities
        ]
        position = np.array(values, dtype=float)
        if not self.is_metamorph_stk:
            # Except for Metamorph STK, invert Y so that stage position
            # coordinates and image pixel coordinates are aligned.
            # FIXME Ask BioFormats team about handling this in the Reader API.
            position *= [-1, 1]
        return position

    @cached_property
    def num_images(self):
        return self.bf_metadata.imageCount

    @cached_property
    def num_tiles(self):
        num_tiles = self.num_images
        # Skip final overview slide in Metamorph Slide Scan data if present.
        if self.is_metamorph_stk and self.has_overview_image:
            num_tiles -= 1
        return num_tiles

    @cached_property
    def format_name(self):
        with self._lock:
            return self.bf_reader.getFormat()

    @cached_property
    def is_metamorph_stk(self):
        return self.format_name == 'Metamorph STK'

    @cached_property
    def has_overview_image(self):
        last_image_name = self.bf_metadata.getImageName(self.num_images - 1)
        return 'overview' in last_image_name.lower()

    def read_image(self, series, channel):
        bf_reader = self.local_bf_reader
        bf_reader.setSeries(series)
        index = bf_reader.getIndex(0, channel, 0)
        byte_array = bf_reader.openBytes(index)
        dtype = self.pixel_dtype
        shape = self.tile_shape
        img = np.frombuffer(byte_array.tostring(), dtype=dtype)
        img = img.reshape(shape)
        return img


@attr.s(frozen=True)
class BioformatsImageReader(metadata.ImageReader):
    reader = attr.ib(validator=attr.validators.instance_of(BioformatsReader))
    series_indices = attr.ib()

    def read(self, image_number, channel):
        series = self.series_indices[image_number]
        img = self.reader.read_image(series, channel)
        return img


def length_as_microns(quantity, name):
    """Return a length quantity's value in microns.

    The `name` of the quantity is used to format a warning message on conversion
    failure.

    """
    value = quantity.value(UNITS.MICROMETER)
    if value is None:
        # Conversion failed, which happens when the unit is "reference
        # frame". Take the bare value as microns, but emit a warning.
        # FIXME Figure out what "reference frame" means and handle this better.
        warnings.warn("No units for {}, assuming micrometers.".format(name))
        value = quantity.value()
    return value.doubleValue()
