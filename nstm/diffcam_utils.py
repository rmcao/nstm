# diffcam_utils.py - Description:
#  Rolling shutter diffuserCam utility functions. Heavily referenced from Nick Antipa's MATLAB code.
# Created by Ruiming Cao on May 22, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import numpy as np
from skimage import io as skimageio
import skimage.transform
from absl import flags


def define_flags():
    flags.DEFINE_string("config", None, "using config files to set hyperparameters.")
    flags.DEFINE_string("raw_path", None, "raw image captured by Diffuser Cam rolling shutter.")
    flags.DEFINE_string("psf_path", None, "PSF path for calibration image.")
    flags.DEFINE_string("save_path", None, "path to save the reconstruction results. ")

    flags.DEFINE_float("readout_time", 27.52, "readout time for each line in microsecond.")
    flags.DEFINE_float("exposure_time", 1320, "exposure time for each frame in microsecond.")

    flags.DEFINE_integer('num_lines_per_forward', 20,
                         'number of readout lines to render for each forward pass.')
    flags.DEFINE_integer('ds', 4, 'downsampling factor for spatial dimension.')

    flags.DEFINE_float("background_int", 103, "background flooring intensity for each pixel.")
    flags.DEFINE_integer("roll_pixels_x", 0, "number of pixels on the raw image to roll in x direction "
                                             "(to put the object at the center).")

    flags.DEFINE_float("annealed_rate", 0.7, "the percentage of epochs to have anneal hash embedding.")
    flags.DEFINE_enum("motion_hash", "separate", ["separate", "combined"],
                      "hash function for motion MLP.")
    flags.DEFINE_integer("object_net_depth", 2, "depth of the object MLP.")
    flags.DEFINE_integer("object_net_width", 128, "width of the object MLP.")
    flags.DEFINE_integer("motion_net_depth", 2, "depth of the motion MLP.")
    flags.DEFINE_integer("motion_net_width", 32, "width of the motion MLP.")
    flags.DEFINE_list("object_hash_base", None, "the base resolution for the hash function of the object.")
    flags.DEFINE_list("object_hash_fine", None, "the fine resolution for the hash function of the object.")
    flags.DEFINE_list("motion_hash_base", None,
                      "the base resolution for the hash function of the motion (zyxt or yxt).")
    flags.DEFINE_list("motion_hash_fine", None,
                      "the fine resolution for the hash function of the motion (zyxt or yxt).")
    flags.DEFINE_string("object_act_fn", 'gelu', "activation function for object MLP.")
    flags.DEFINE_string("motion_act_fn", 'elu', "activation function for motion MLP.")

    flags.DEFINE_float("nonneg_reg_w", 1e-5, "weight for non-negative regularization.")
    flags.DEFINE_float("lr_init_object", 1e-3, "The initial learning rate for object net.")
    flags.DEFINE_float("lr_init_motion", 5e-5, "The initial learning rate for motion net.")
    flags.DEFINE_integer("batch_size", 1, "batch size for training.")
    flags.DEFINE_integer("num_epoch", 100, "number of epoches for reconstruction.")
    flags.DEFINE_integer("update_every_object", 1, "number of steps for each update of object network.")
    flags.DEFINE_integer("update_every_motion", 1, "number of steps for each update of motion network.")

    flags.DEFINE_integer("save_every", 1000,
                         "the number of steps to save a checkpoint.")
    flags.DEFINE_integer("print_every", 10,
                         "the number of steps between reports to tensorboard.")
    flags.DEFINE_integer(
        "render_every", 20, "the number of steps to render a test image,"
                            "better to be x00 for accurate step time record.")


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

