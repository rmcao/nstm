# sim3d_utils.py - Description:
#  Utility functions for 3D SIM reconstruction. Some functions are heavily referenced from the cuda-accelerated
#  three-beam SIM reconstruction code (https://github.com/scopetools/cudasirecon).
# Created by Ruiming Cao on Apr 07, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from typing import Tuple, Union, List, Sequence
from absl import flags
import numpy as np
import jax.numpy as jnp
from sklearn import linear_model
from scipy import stats
from flax.struct import dataclass

from nstm.utils import SystemParameters, SystemParameters3D, notch_filter


def define_flags():
    """Define flags."""
    flags.DEFINE_bool("eval_only", False,
                      "Output the reconstructed frames and save motion map with existing model.")
    flags.DEFINE_string("config", None, "using config files to set hyperparameters.")
    flags.DEFINE_integer("seed", 0, "Seed for random number generators.")

    # data parameters
    flags.DEFINE_string("data_dir", None, "input data directory.")
    flags.DEFINE_string("raw_path", None, "raw 3D SIM image path. regex for multiple files of time series acquisition."
                                          "Supported formats: .tif, .czi, and .npz. The .npz file has to be generated "
                                          "by process_raw_images.py script or follow the same organization. When a "
                                          ".npz file is provided, no otf_path and meta_path are needed, and patch_json "
                                          "should not be used.")
    flags.DEFINE_string("save_path", None, "path to save the reconstruction results. "
                                           "This will overwrite the save path in full2patch.json file.")

    # these parameters will be skipped if a .npz file is provided in the raw_path
    flags.DEFINE_string("otf_path", None, "OTF path for 3D SIM. regex for multiple files.")
    flags.DEFINE_string("meta_path", None, "path for metadata of 3D SIM.")
    flags.DEFINE_string("patch_json", None, "path to full-to-patches csv file.")
    flags.DEFINE_integer("patch_ind", None, "patch index for the current run.")
    flags.DEFINE_list("coord_start", None, "the starting coordinate of the FOV in zyx (3D) or yx (2D).")
    flags.DEFINE_list("patch_dim", None, "the dimension of the FOV in zyx (3D) or yx (2D).")
    flags.DEFINE_float("background_int", 100, "background flooring intensity for each pixel.")
    flags.DEFINE_integer("num_stack", 1,
                         "number of image stacks to use for the reconstruction of 3D SIM.")
    flags.DEFINE_integer("starting_stack", 0, "index for the first image stack used in the "
                         "reconstruction for 3D SIM.")

    # optical system parameters
    flags.DEFINE_float("ps", 0.092, "pixel size in microns.")
    flags.DEFINE_float("dz", 0.15, "axial step size in microns.")
    flags.DEFINE_float("zoomfact", 2, "zoom factor on lateral dimensions.")
    flags.DEFINE_float("na", 1.0, "numerical aperture.")
    flags.DEFINE_float("wavelength", 0.515, "emission light wavelength in microns.")
    flags.DEFINE_float("wavelength_exc", 0.488, "excitation light wavelength in microns.")
    flags.DEFINE_float("ri_medium", 1.33, "refractive index of the medium.")
    flags.DEFINE_integer("padding", 0, "lateral dimension padding size for the image.")
    flags.DEFINE_integer("padding_z", 0, "axial dimension padding size for the image.")

    # 3D SIM parameters
    flags.DEFINE_float("ps_otf", 0.092, "pixel size in microns for OTF.")
    flags.DEFINE_float("dz_otf", 0.1, "axial step size in microns for OTF.")
    flags.DEFINE_list("dim_otf_zx", None,
                      "The dimension of the radial-averaged OTF for 3D SIM, on z-x, in pixels.")
    flags.DEFINE_integer("ndirs", 3, "number of directions for 3D SIM.")
    flags.DEFINE_integer("nphases", 5, "number of phase shifts for 3D SIM.")
    flags.DEFINE_list("line_spacing", None,
                      "The line spacing of the SIM pattern, on each direction, in pixels.")
    flags.DEFINE_list("k0angles", None,
                      "The k0 angles of the SIM pattern, on each direction, in radians.")
    flags.DEFINE_float("band0_grad_reduction", 0.0,
                       "the reduction factor for the gradient of band 0.")

    # spacetime model parameters
    flags.DEFINE_bool("fast_mode", True, "whether to use fast mode (recommended) for 3D reconstruction.")
    flags.DEFINE_string("precision", "float32", "numerical precision for the computation.")
    flags.DEFINE_enum("motion_hash", "combined", ["separate", "combined"],
                      "hash function for motion MLP.")
    flags.DEFINE_integer("object_net_depth", 2, "depth of the  object MLP.")
    flags.DEFINE_integer("object_net_width", 128, "width of the object MLP.")
    flags.DEFINE_integer("motion_net_depth", 2, "depth of the motion MLP.")
    flags.DEFINE_integer("motion_net_width", 32, "width of the motion MLP.")
    flags.DEFINE_string("object_out_activation", None,
                        "activation function for the output of object MLP.")
    flags.DEFINE_float("annealed_rate", 0.8,
                       "the percentage of epochs to have anneal hash embedding.")
    flags.DEFINE_list("object_hash_base", None,
                      "the base resolution for the hash function of the object.")
    flags.DEFINE_list("object_hash_fine", None,
                      "the fine resolution for the hash function of the object.")
    flags.DEFINE_list("motion_hash_base", None,
                      "the base resolution for the hash function of the motion (zyxt or yxt).")
    flags.DEFINE_list("motion_hash_fine", None,
                      "the fine resolution for the hash function of the motion (zyxt or yxt).")
    flags.DEFINE_float("object_hash_boundary_x", 1.0,
                       "the boundary extension ratio of the object hash grid. 0 for no extension from the image "
                       "dimension. 1 (default) for 50% extension on each side.")

    # loss parameters
    flags.DEFINE_integer("l2_loss_margin", 2,
                         "image margin excluded for the computation of l2 loss.")
    flags.DEFINE_float("nonneg_reg_w", 0, "weight for non-negative regularization.")

    # reconstruction parameters
    flags.DEFINE_integer("batch_size", 3, "batch size for training.")
    flags.DEFINE_integer("num_epoch", 1000, "number of epoches for reconstruction.")
    flags.DEFINE_float("lr_init_object", 1e-3, "The initial learning rate.")
    flags.DEFINE_float("lr_init_motion", 1e-5, "The initial learning rate.")
    flags.DEFINE_float("lr_init_modamp", 5e-3, "The initial learning rate for modulation amplitude.")
    flags.DEFINE_float("lr_final", 5e-8, "The final learning rate.")
    flags.DEFINE_float("lr_decay_motion", 0.1,
                       "The decay of motion network learning rate after all epochs.")
    flags.DEFINE_float("lr_decay_object", 0.1,
                       "The decay of object network learning rate after all epochs.")
    flags.DEFINE_integer(
        "lr_delay_steps", 0, "The number of steps at the beginning of "
                             "training to reduce the learning rate by lr_delay_mult")
    flags.DEFINE_float("modamp_delay", None,
                       "The percentage of epochs to delay the update of modulation amplitude.")
    flags.DEFINE_integer("update_every_object", 1,
                         "number of steps for each update of object network.")
    flags.DEFINE_integer("update_every_motion", 1,
                         "number of steps for each update of motion network.")
    flags.DEFINE_integer("save_every", 1000,
                         "the number of steps to save a checkpoint.")
    flags.DEFINE_integer("print_every", 10,
                         "the number of steps between reports to tensorboard.")
    flags.DEFINE_integer(
        "render_every", 50, "the number of steps to render a test image,"
                            "better to be x00 for accurate step time record.")


@dataclass
class SIMParameter3D:
    OTF: jnp.ndarray
    nphases: int
    ndirs: int
    k0angles: Tuple[float]
    line_spacing: Tuple[float]
    starting_phases: Tuple[Tuple[float]]
    origin_pixel_offset_yx: Tuple[float]


def generate_sinusoidal(param: Union[SystemParameters, SystemParameters3D],
                        k0angle: float,
                        k0mag: float,
                        phase: float,
                        order: int,
                        origin_pixel_offset_yx: Tuple[float] = None):
    if 'dim_yx' in param.__annotations__:
        dim_yx = (param.dim_yx[0] + param.padding_yx[0] * 2, param.dim_yx[1] + param.padding_yx[1] * 2)
        dx = param.pixel_size
    elif 'dim_zyx' in param.__annotations__:
        dim_yx = (param.dim_zyx[1] + param.padding_zyx[1] * 2, param.dim_zyx[2] + param.padding_zyx[2] * 2)
        dx = param.pixel_size
    else:
        raise ValueError('Wrong param type')

    if origin_pixel_offset_yx is None:
        origin_pixel_offset_yx = (0, 0)

    k0x = k0mag * order * np.cos(k0angle)
    k0y = k0mag * order * np.sin(k0angle)

    xlin = (np.arange(dim_yx[1]) - dim_yx[1]/2 + origin_pixel_offset_yx[1]) * dx
    ylin = (np.arange(dim_yx[0]) - dim_yx[0]/2 + origin_pixel_offset_yx[0]) * dx

    ys, xs = np.meshgrid(ylin, xlin, indexing='ij')

    I_sin = np.cos((xs * k0x + ys * k0y) * 2 * np.pi + phase * order)
    return I_sin


def generate_exp(param: Union[SystemParameters, SystemParameters3D],
                 k0angle: float,
                 k0mag: float,
                 phase: float,
                 order: int,
                 origin_pixel_offset_yx: Tuple[float] = None):
    if 'dim_yx' in param.__annotations__:
        dim_yx = (param.dim_yx[0] + param.padding_yx[0] * 2, param.dim_yx[1] + param.padding_yx[1] * 2)
        dx = param.pixel_size
    elif 'dim_zyx' in param.__annotations__:
        dim_yx = (param.dim_zyx[1] + param.padding_zyx[1] * 2, param.dim_zyx[2] + param.padding_zyx[2] * 2)
        dx = param.pixel_size
    else:
        raise ValueError('Wrong param type')

    if order == 0:
        return np.ones(dim_yx, dtype=np.complex128)

    if origin_pixel_offset_yx is None:
        origin_pixel_offset_yx = (0, 0)

    k0x = k0mag * order * np.cos(k0angle)
    k0y = k0mag * order * np.sin(k0angle)

    xlin = (np.arange(dim_yx[1]) - dim_yx[1]/2 + origin_pixel_offset_yx[1]) * dx
    ylin = (np.arange(dim_yx[0]) - dim_yx[0]/2 + origin_pixel_offset_yx[0]) * dx

    ys, xs = np.meshgrid(ylin, xlin, indexing='ij')

    I_exp = np.exp(1.0j * ((xs * k0x + ys * k0y) * 2 * np.pi + phase * order))
    return I_exp


def rad_avg_OTF_expansion(rad_avg_OTF: np.ndarray,
                          img_param: SystemParameters3D,
                          otf_param: SystemParameters3D,
                          kxy_shift: Tuple[float] = None,
                          freq_cutoff: bool = False,
                          order: int = 0,
                          with_padding: bool = False):
    assert(rad_avg_OTF.shape[0] == otf_param.dim_zyx[2]//2 + 1)

    if with_padding:
        dim_zyx = (img_param.dim_zyx[0] + img_param.padding_zyx[0] * 2,
                   img_param.dim_zyx[1] + img_param.padding_zyx[1] * 2,
                   img_param.dim_zyx[2] + img_param.padding_zyx[2] * 2)
    else:
        dim_zyx = img_param.dim_zyx

    kxlin = np.fft.fftfreq(dim_zyx[2], img_param.pixel_size)
    kylin = np.fft.fftfreq(dim_zyx[1], img_param.pixel_size)
    kzlin = np.fft.fftfreq(dim_zyx[0], img_param.pixel_size_z)
    dkz = 1 / dim_zyx[0] / img_param.pixel_size_z

    if kxy_shift:
        kxlin -= kxy_shift[0]
        kylin -= kxy_shift[1]

    nzotf, nxotf = rad_avg_OTF.shape[1], rad_avg_OTF.shape[0]
    dkz_otf = 1 / nzotf / otf_param.pixel_size_z
    dkr_otf = 1 / (nxotf-1) / 2 / otf_param.pixel_size

    ky, kx = np.meshgrid(kylin, kxlin, indexing='ij')
    kr = np.sqrt(ky**2 + kx**2)
    krindex = kr / dkr_otf
    kzindex = kzlin / dkz_otf
    kzindex = np.where(kzindex >= 0, kzindex, kzindex + nzotf)
    krindex = np.tile(krindex[np.newaxis, :, :], (dim_zyx[0], 1, 1))
    kzindex = np.tile(kzindex[:, np.newaxis, np.newaxis], (1, dim_zyx[1], dim_zyx[2]))

    irindex = np.minimum(np.floor(krindex).astype(int), nxotf - 1)
    irindex_ = np.minimum(irindex + 1, nxotf - 1)
    izindex = np.minimum(np.floor(kzindex).astype(int), nzotf - 1)
    izindex_ = np.minimum(izindex + 1, nzotf - 1)

    ar = krindex - irindex
    az = kzindex - izindex

    otf_out = (1-ar) * (rad_avg_OTF[irindex, izindex] * (1-az) + rad_avg_OTF[irindex, izindex_] * az) + \
              ar * (rad_avg_OTF[irindex_, izindex] * (1-az) + rad_avg_OTF[irindex_, izindex_] * az)

    if freq_cutoff:
        krdist_cutoff = img_param.na * 2 / img_param.wavelength
        if krdist_cutoff > 0.5 / img_param.pixel_size:
            krdist_cutoff = 0.5 / img_param.pixel_size

        kzdist_cutoff = np.ceil((1 - np.cos(img_param.na/img_param.RI_medium)) / img_param.wavelength / dkz)

        if order == 1:
            kzdist_cutoff = kzdist_cutoff * 2
        elif order == 2:
            kzdist_cutoff = kzdist_cutoff * 1.3
        if kzdist_cutoff > dim_zyx[0]/2:
            kzdist_cutoff = dim_zyx[0]/2 - 1

        otf_out = np.where((kr[np.newaxis, :, :] <= krdist_cutoff) & (np.abs(kzlin[:, np.newaxis, np.newaxis] / dkz) <= kzdist_cutoff),
                           otf_out, np.zeros_like(otf_out))

    return otf_out


def otf_support_mask(param: SystemParameters3D,
                     otf_param: SystemParameters3D,
                     sim_param: SIMParameter3D,
                     otf,
                     otf_cutoff: float = 1e-8):
    """Output a 3D binary mask for possible supported regions of 3D SIM."""

    otf_valid_region = np.abs(rad_avg_OTF_expansion(otf[:, :, 0], param, otf_param, (0, 0))) > otf_cutoff
    for i_k0 in range(sim_param.ndirs):
        for i_o in range(1, 3):
            k0x = 1 / sim_param.line_spacing[i_k0] * i_o * np.cos(sim_param.k0angles[i_k0])
            k0y = 1 / sim_param.line_spacing[i_k0] * i_o * np.sin(sim_param.k0angles[i_k0])

            otf_valid_region += np.abs(
                rad_avg_OTF_expansion(otf[:, :, i_o], param, otf_param, (k0x, k0y), with_padding=False)) > otf_cutoff
            otf_valid_region += np.abs(
                rad_avg_OTF_expansion(otf[:, :, i_o], param, otf_param, (-k0x, -k0y), with_padding=False)) > otf_cutoff

    otf_valid_region_rfft = (otf_valid_region[:, :, :int(param.dim_zyx[2] // 2 + 1)] > 0).astype(np.float32)  # for rfft

    # otf_valid_region[0, 128, 128] = True
    otf_valid_region = (otf_valid_region > 0).astype(np.float32)
    return otf_valid_region, otf_valid_region_rfft


def make_overlaps(band1, band2, order1, order2, otf,
                  img_param: SystemParameters3D, otf_param: SystemParameters3D,
                  k0angle: float, k0mag: float, phase2: float, normalize_otf=True):
    """makeOverlaps0Kernel & makeOverlaps1Kernel"""
    if order1 != 0:
        raise NotImplementedError

    k0x_2 = k0mag * order2 * np.cos(k0angle)
    k0y_2 = k0mag * order2 * np.sin(k0angle)

    dim_zyx = (img_param.dim_zyx[0], img_param.dim_zyx[1], img_param.dim_zyx[2])

    krdist_cutoff = img_param.na * 2 / img_param.wavelength
    kzdist_cutoff = np.ceil((1 - np.cos(img_param.na / img_param.RI_medium)) / img_param.wavelength)

    kxlin = np.fft.fftfreq(dim_zyx[2], img_param.pixel_size)
    kylin = np.fft.fftfreq(dim_zyx[1], img_param.pixel_size)
    kzlin = np.fft.fftfreq(dim_zyx[0], img_param.pixel_size_z)
    ky, kx = np.meshgrid(kylin, kxlin, indexing='ij')

    otf1 = rad_avg_OTF_expansion(otf[:, :, order1], img_param, otf_param, kxy_shift=(0, 0), order=order1, with_padding=False)
    otf12 = rad_avg_OTF_expansion(otf[:, :, order2], img_param, otf_param, kxy_shift=(k0x_2, k0y_2), order=order2, with_padding=False)
    otf2 = rad_avg_OTF_expansion(otf[:, :, order2], img_param, otf_param, kxy_shift=(0, 0), order=order2, with_padding=False)
    otf21 = rad_avg_OTF_expansion(otf[:, :, order1], img_param, otf_param, kxy_shift=(-k0x_2, -k0y_2), order=order1, with_padding=False)

    valid_mask = np.zeros(dim_zyx)
    valid_mask[((np.sqrt(ky**2 + kx**2) <= krdist_cutoff) &
                ((np.sqrt((ky-k0y_2)**2 + (kx-k0x_2)**2) <= krdist_cutoff) |
                 (np.sqrt((ky+k0y_2)**2 + (kx+k0x_2)**2) <= krdist_cutoff)))[np.newaxis] &
               (np.abs(kzlin)[:,np.newaxis,np.newaxis] <= kzdist_cutoff)] = 1
    valid_mask_1 = ((np.sqrt(ky**2 + kx**2) <= krdist_cutoff) & (np.sqrt((ky-k0y_2)**2 + (kx-k0x_2)**2) <= krdist_cutoff))[np.newaxis] &\
               (np.abs(kzlin)[:,np.newaxis,np.newaxis] <= kzdist_cutoff)
    valid_mask_2 = ((np.sqrt(ky**2 + kx**2) <= krdist_cutoff) & (np.sqrt((ky+k0y_2)**2 + (kx+k0x_2)**2) <= krdist_cutoff))[np.newaxis] &\
               (np.abs(kzlin)[:,np.newaxis,np.newaxis] <= kzdist_cutoff)

    # each overlap region is bounded by two OTFs
    valid_mask_1 *= (np.abs(otf1) >= 1e-6) * (np.abs(otf12) >= 1e-6)
    valid_mask_2 *= (np.abs(otf2) >= 1e-6) * (np.abs(otf21) >= 1e-6)

    fact1 = otf12
    fact2 = otf21
    if normalize_otf:
        fact1 /= (np.sqrt(np.abs(otf1)**2 + np.abs(otf12)**2) + 1e-8)
        fact2 /= (np.sqrt(np.abs(otf2)**2 + np.abs(otf21)**2) + 1e-8)

    overlap1 = band1 * fact1 * valid_mask_1
    overlap2 = band2 * fact2 * valid_mask_2

    # to real space
    overlap1 = np.fft.ifftn(overlap1)
    overlap2 = np.fft.ifftn(overlap2)

    # frequency shift overlap2 by (order2-order1)*k0
    overlap2 *= generate_exp(img_param, k0angle, k0mag, phase2, order2)

    # output in real space
    return overlap1, overlap2


def get_modamp(overlap1: np.ndarray, overlap2: np.ndarray, crop_boundary_zyx: Sequence[int] = None,
               intercept: bool = True, drop_half: bool = False):
    """Estimate for the relative modulation amplitude and starting phase for overlap2 w.r.t. overlap1 (from order 0)."""
    if crop_boundary_zyx:
        dim_zyx = overlap1.shape
        overlap1 = overlap1[crop_boundary_zyx[0]:dim_zyx[0]-crop_boundary_zyx[0],
                            crop_boundary_zyx[1]:dim_zyx[1]-crop_boundary_zyx[1],
                            crop_boundary_zyx[2]:dim_zyx[2]-crop_boundary_zyx[2]]
        overlap2 = overlap2[crop_boundary_zyx[0]:dim_zyx[0]-crop_boundary_zyx[0],
                            crop_boundary_zyx[1]:dim_zyx[1]-crop_boundary_zyx[1],
                            crop_boundary_zyx[2]:dim_zyx[2]-crop_boundary_zyx[2]]

    overlap1 = overlap1.flatten()
    overlap2 = overlap2.flatten()

    if drop_half:
        ind = np.argsort(np.abs(overlap2))
        overlap1 = overlap1[ind][len(ind)//2:]
        overlap2 = overlap2[ind][len(ind)//2:]
    # return overlap1, overlap2
    complex_reg = stats.linregress(overlap2, overlap1)
    starting_phase = np.angle(complex_reg.slope)
    if intercept is False:
        real_reg = linear_model.LinearRegression(fit_intercept=False).fit(np.abs(overlap2.reshape(-1,1)), np.abs(overlap1))
        # real_reg = svm.LinearSVR(epsilon=0., C=10, max_iter=100).fit(np.abs(overlap2.reshape(-1,1)), np.abs(overlap1))
        amp = real_reg.coef_[0]
    else:
        amp = np.abs(complex_reg.slope)
    return starting_phase, amp


def gen_dampen_order0_mask(param: SystemParameters3D, inverted=False):
    dim_zyx = (param.dim_zyx[0] + param.padding_zyx[0] * 2,
               param.dim_zyx[1] + param.padding_zyx[1] * 2,
               param.dim_zyx[2] + param.padding_zyx[2] * 2)

    frdistcutoff = 2.0 * param.na / param.wavelength
    fzdistcutoff = np.ceil((1.0 - np.cos(param.na / param.RI_medium)) / param.wavelength)

    fxlin = np.fft.fftfreq(dim_zyx[2], param.pixel_size)
    fylin = np.fft.fftfreq(dim_zyx[1], param.pixel_size)
    fzlin = np.fft.fftfreq(dim_zyx[0], param.pixel_size_z)

    fyy, fxx = np.meshgrid(fylin, fxlin, indexing='ij')
    fr_obj = np.sqrt(fxx ** 2 + fyy ** 2)

    out = (fr_obj / frdistcutoff)[np.newaxis, ...]**2 + np.abs(fzlin / fzdistcutoff)[:, np.newaxis, np.newaxis]**3
    out = np.where(out > 1, 1, out)

    if inverted:
        out = 1 / (out + 2e-1)
    return out


def separate_bands(imgs: np.ndarray, # [nphases, z, y, x]
                   nphases=5, norders=3, out_positive_bands=False):
    nphases_, dim_z, dim_y, dim_x = imgs.shape
    assert(nphases_ == nphases)

    f_imgs = np.fft.fft2(imgs, axes=(-2, -1))

    # create sep matrix
    phi = 2 * np.pi / nphases
    sepmatrix = np.zeros((norders * 2 - 1, nphases), dtype=np.complex128)

    for j in range(nphases):
        sepmatrix[0, j] = 1.0
        for order in range(1, norders):
            sepmatrix[2*order - 1, j] = np.cos(j * order * phi)
            sepmatrix[2*order, j] = np.sin(j * order * phi)

    print(sepmatrix.real)

    out = np.zeros((norders * 2 -1, dim_z, dim_y, dim_x), dtype=np.complex128)
    for i in range(norders * 2 -1):
        for j in range(nphases):
            out[i] += sepmatrix[i, j] * f_imgs[j]

    # rescale the output values back to the input level
    out = out / 5.0

    if out_positive_bands:
        bandplus_img = np.zeros((3, dim_z, dim_y, dim_x), dtype=np.complex128)
        bandplus_img[0] = out[0]
        bandplus_img[1] = out[1] + 1.0j * out[2]
        bandplus_img[2] = out[3] + 1.0j * out[4]

        out = np.fft.ifft2(bandplus_img, axes=(-2, -1))
    else:
        out = np.fft.ifft2(out, axes=(-2, -1))

    return out


def estimate_mod_illum(bandplus_img: np.ndarray,  # [ndirs, band, z, y, x]
                       otf: List[np.ndarray],
                       img_param: SystemParameters3D,
                       otf_param: SystemParameters3D,
                       ndirs: int,
                       k0angles: Sequence[float],
                       line_spacing: Sequence[float],
                       crop_boundary_zyx: Sequence[int],
                       noisy=True):
    f_bandplus_img = np.fft.fftn(bandplus_img, axes=(-3, -2, -1))

    phase_dir_order = np.zeros((ndirs, 3))
    amp_dir_order = np.ones((ndirs, 3))

    band0_amp = np.mean(np.abs(bandplus_img[:, 0]), axis=(-3, -2, -1))
    band0_amp = band0_amp / band0_amp[0]

    for i_k0angle in range(ndirs):
        for order_2 in range(1, 3):
            overlap1, overlap2 = make_overlaps(f_bandplus_img[i_k0angle, 0], f_bandplus_img[i_k0angle, order_2],
                                               0, order_2, otf[i_k0angle], img_param, otf_param,
                                               k0angle=k0angles[i_k0angle],
                                               k0mag=1 / line_spacing[i_k0angle],
                                               phase2=0,
                                               normalize_otf=True)
            if noisy:
                phase, amp = get_modamp(overlap1, overlap2, crop_boundary_zyx, intercept=False, drop_half=False)
            else:
                phase, amp = get_modamp(overlap1, overlap2, crop_boundary_zyx, intercept=True, drop_half=False)
            phase_dir_order[i_k0angle, order_2] = phase / order_2
            amp_dir_order[i_k0angle, order_2] = 1 / amp

    amp_dir_order = amp_dir_order * band0_amp[:, np.newaxis]

    return phase_dir_order, amp_dir_order


def get_otf(bandplus_img: np.ndarray,  # [ndirs, nphases, z, y, x]
            otf: List[np.ndarray],
            img_param: SystemParameters3D,
            otf_param: SystemParameters3D,
            ndirs: int,
            nphases: int,
            k0angles: Sequence[float],
            line_spacing: Sequence[float],
            crop_boundary_zyx: Sequence[int],
            noisy=True,
            notch=False,
            notch_width=0.5):
    phase, amp = estimate_mod_illum(bandplus_img, otf, img_param, otf_param, ndirs, k0angles, line_spacing,
                                    crop_boundary_zyx=crop_boundary_zyx, noisy=noisy)
    print(phase, flush=True)
    print(amp, flush=True)

    otf_3d = [[rad_avg_OTF_expansion(otf[i_k0angle][:, :, o], img_param, otf_param, freq_cutoff=False,
                                     order=o, with_padding=True) *
               amp[i_k0angle][o] for o in range(3)] for i_k0angle in range(ndirs)]

    if notch:
        for i_k0angle in range(ndirs):
            for o in range(3):
                kzlin = np.fft.fftfreq(otf_param.dim_zyx[0], otf_param.pixel_size_z)
                kz_index = np.argmax(otf[i_k0angle][:, :, o])
                otf_3d[i_k0angle][o] *= notch_filter(img_param, o, d=0.8, w=notch_width, kz_offset=kzlin[kz_index],
                                                     inverted=True)

    sim_param_new = SIMParameter3D(OTF=np.array(otf_3d), nphases=nphases, ndirs=ndirs,
                                   k0angles=k0angles, line_spacing=line_spacing,
                                   starting_phases=phase, origin_pixel_offset_yx=(0, 0))

    return sim_param_new, phase, amp