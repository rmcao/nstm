# diffcam_utils.py - Description:
#  Rolling shutter diffuserCam utility functions. Heavily referenced from Nick Antipa's MATLAB code.
# Created by Ruiming Cao on May 22, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import glob
import numpy as np
from skimage import io as skimageio
import skimage.transform


def load_data_psf(raw_path, psf_path, background, downsample=8):
    psf = np.maximum(skimageio.imread(psf_path).astype(np.float32) - background, 0)
    psf = skimage.transform.downscale_local_mean(psf, (downsample, downsample, 1))
    h = psf / np.max(np.sum(psf, axis=(0, 1)))

    raw = np.maximum(skimageio.imread(raw_path).astype(np.float32) - background, 0)
    raw = skimage.transform.downscale_local_mean(raw, (downsample, downsample, 1))
    raw = raw / np.max(raw)

    return raw, h


def gen_indicator(dims, nlines, pad2d, downsample_t=True):
    indicator = np.zeros(dims, dtype=bool)
    counter = 0
    layer = np.ones(dims, dtype=bool)

    while np.sum(layer):
        layer = np.zeros(dims, dtype=bool)
        counter += 1
        top_min = max(0, counter - nlines)
        top_max = min(counter, int(dims[0] // 2))
        bot_min = max(dims[0] - counter, int(dims[0] // 2))
        bot_max = min(dims[0], dims[0] - counter + nlines)

        layer[top_min:top_max, :] = True
        layer[bot_min:bot_max, :] = True

        if counter == 1:
            indicator = pad2d(layer)
        elif np.sum(layer):  # If empty, don't keep
            indicator = np.dstack((indicator, pad2d(layer)))

        if counter > dims[0]:
            raise ValueError('while loop is going on too long in hsvid_crop')

    indicator = indicator[:, :, :int(indicator.shape[2] // 2) * 2].astype(np.float32)

    if downsample_t:
        # downsample time by a factor of two
        indicator = 0.5 * indicator[:, :, ::2] + 0.5 * indicator[:, :, 1::2]

        if indicator.shape[2] % 2 != 0:
            indicator[:, :, -2] += indicator[:, :, -1]
            indicator = indicator[:, :, :-1]  # Make it even

    return indicator

