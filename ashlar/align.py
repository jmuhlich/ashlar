import itertools
import numbers
import threading
import attr
import attr.validators as av
import numpy as np
import scipy.ndimage as ndimage
import skimage.feature
import skimage.filters
import skimage.restoration

from . import geometry
from .util import cached_property


@attr.s(frozen=True)
class PlaneAlignment(object):
    shift = attr.ib(validator=av.instance_of(geometry.Vector))
    error = attr.ib()


@attr.s(frozen=True)
class EdgeTileAlignment(object):
    plane_alignment = attr.ib(validator=av.instance_of(PlaneAlignment))
    tile_index_1 = attr.ib()
    tile_index_2 = attr.ib()

    def __attrs_post_init__(self):
        # Normalize so that tile_index_1 < tile_index_2.
        if self.tile_index_1 > self.tile_index_2:
            t1 = self.tile_index_1
            t2 = self.tile_index_2
            new_shift = -self.plane_alignment.shift
            new_alignment = attr.evolve(self.plane_alignment, shift=new_shift)
            object.__setattr__(self, 'tile_index_1', t2)
            object.__setattr__(self, 'tile_index_2', t1)
            object.__setattr__(self, 'plane_alignment', new_alignment)

    @cached_property
    def tile_indexes(self):
        return (self.tile_index_1, self.tile_index_2)

    def get_shift(self, index):
        """Return the shift from the "perspective" of tile `index`."""
        if index == self.tile_index_1:
            return -self.plane_alignment.shift
        elif index == self.tile_index_2:
            return self.plane_alignment.shift
        else:
            raise ValueError("Invalid tile index")

    @property
    def shift(self):
        return self.plane_alignment.shift

    @property
    def error(self):
        return self.plane_alignment.error


def register_planes(plane1, plane2):
    if plane1.pixel_size != plane2.pixel_size:
        raise ValueError("planes have different pixel sizes")
    if plane1.bounds.shape != plane2.bounds.shape:
        raise ValueError("planes have different shapes")
    if plane1.bounds.area == 0:
        raise ValueError("planes are empty")
    shift_pixels, error = register(plane1.image, plane2.image)
    shift = geometry.Vector.from_ndarray(shift_pixels) * plane1.pixel_size
    shift_adjusted = shift + (plane1.bounds.vector1 - plane2.bounds.vector1)
    return PlaneAlignment(shift_adjusted, error)


def register(img1, img2, upsample_factor=10):
    """Return translation shift from img2 to img2 and an error metric.

    This function wraps skimage registration to apply our conventions and
    enhancements. We pre-whiten the input images, always provide fourier-space
    input images, resolve the phase confusion problem, and report an improved
    (to us) error metric.

    """
    img1w = whiten(img1)
    img2w = whiten(img2)
    img1_f = np.fft.fft2(img1w)
    img2_f = np.fft.fft2(img2w)
    img1w = img1w.real
    img2w = img2w.real
    shift, _, _ = skimage.feature.register_translation(
        img1_f, img2_f, upsample_factor, 'fourier'
    )
    # At this point we may have a shift in the wrong quadrant since the FFT
    # assumes the signal is periodic. We test all four possibilities and return
    # the shift that gives the highest direct correlation (sum of products).
    shape = np.array(img1.shape)
    shift_pos = (shift + shape) % shape
    shift_neg = shift_pos - shape
    shifts = list(itertools.product(*zip(shift_pos, shift_neg)))
    correlations = [
        np.sum(img1w * ndimage.shift(img2w, np.rint(s)))
        for s in shifts
    ]
    idx = np.argmax(correlations)
    shift = np.array(shifts[idx])
    correlation = correlations[idx]
    total_amplitude = np.linalg.norm(img1w) * np.linalg.norm(img2w)
    if correlation > 0 and total_amplitude > 0:
        error = -np.log(correlation / total_amplitude)
    else:
        error = np.inf
    return shift, error


# Pre-calculate the Laplacian operator kernel. We'll always be using 2D images.
_laplace_kernel = skimage.restoration.uft.laplacian(2, (3, 3))[1]

def whiten(img, sigma=0.0):
    """Return a spectrally whitened copy of an image with optional smoothing.

    Uses Laplacian of Gaussian with the given sigma. Returns a complex64 output
    with the whitened image in the real component and zero in the imaginary
    component. (This allows the result to be used directly in FFT operations)

    """
    output = np.empty_like(img, dtype=np.complex64)
    img = skimage.img_as_float(img)
    if sigma == 0:
        ndimage.convolve(img, _laplace_kernel, output.real)
    else:
        ndimage.gaussian_laplace(img, sigma, output=output.real)
    output.imag[:] = 0
    return output


def ishift(input, shift):
    sy = input.shape[0] - shift[0]
    sx = input.shape[1] - shift[1]
    if shift[0] >= 0:
        iy1 = 0
        iy2 = sy
        oy1 = shift[0]
        oy2 = input.shape[0]
    else:
        iy1 = shift[0]
        iy2 = input.shape[0]
        oy1 = 0
        oy2 = sy
    if shift[1] >= 0:
        ix1 = 0
        ix2 = sx
        ox1 = shift[1]
        ox2 = input.shape[1]
    else:
        ix1 = shift[1]
        ix2 = input.shape[1]
        ox1 = 0
        ox2 = sx
    output = np.zeros_like(input)
    output[oy1:oy2, ox1:ox2] = input[oy1:oy2, ox1:ox2]
    return output
