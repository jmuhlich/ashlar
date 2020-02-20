import numpy as np
import skimage
from ashlar import reg

def rg(img1, img2):
    assert img1.shape == img2.shape
    assert img1.dtype == img2.dtype
    out = np.empty(img1.shape + (3,), dtype=np.float32)
    out[...,0] = skimage.img_as_float32(img1)
    out[...,1] = skimage.img_as_float32(img2)
    out[...,2] = 0
    return out

reader1 = reg.BioformatsReader(
    'input/mmo12_ashlar_test/Original_180328/Set 2/Scan 20x obj 3.scan'
)
reader1.metadata.positions
reader1.metadata._positions *= [-1, 1]
aligner1 = reg.EdgeAligner(
    reader1, 0, verbose=True, max_shift=30, filter_sigma=1.0
)
aligner1.run()

reader2 = reg.BioformatsReader(
    'input/mmo12_ashlar_test/Rescanned_180328/Set 2/Scan 20x obj 3.scan'
)
reader2.metadata.positions
reader2.metadata._positions *= [-1, 1]
aligner2 = reg.LayerAligner(
    reader2, aligner1, 0, verbose=True, filter_sigma=1.0, max_shift=30,
    max_rotation_dev=1
)
aligner2.run()

mosaic1 = reg.Mosaic(
    aligner1, aligner1.mosaic_shape, '', channels=[0], verbose=True
)
img1, = mosaic1.run(mode='return')
mosaic2 = reg.Mosaic(
    aligner2, aligner1.mosaic_shape, '', channels=[0], verbose=True
)
img2, = mosaic2.run(mode='return')
img = np.dstack([img1, img2])
