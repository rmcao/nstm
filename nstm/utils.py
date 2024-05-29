# -*- coding: utf-8 -*-
"""General utility functions and dynamic simulation tools used in neural space-time model paper."""

from typing import Tuple, Union, List
from os import path
import yaml
import numpy as np
from PIL import Image
from flax.struct import dataclass
from skimage import data, transform
import skimage
import imageio
from scipy.stats import norm

from calcil.physics.wave_optics import propKernelNumpy, genGridNumpy


def update_flags(args):
    """Update the flags in `args` with the contents of the config YAML file.

    Copied from https://github.com/google-research/google-research/blob/master/jaxnerf/nerf/utils.py"""

    # check if args.config ends with .yaml
    if not args.config.endswith(".yaml"):
        pth = path.join('./examples/configs/', args.config + ".yaml")
    else:
        pth = args.config
    with open(pth, mode="r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Only allow args to be updated if they already exist.
    invalid_args = list(set(configs.keys()) - set(dir(args)))
    if invalid_args:
        raise ValueError(f"Invalid args {invalid_args} in {pth}.")
    args._set_attributes(**configs)


@dataclass
class SystemParameters:
    """Imaging system parameters for 2D imaging systems.

    Args:
        """
    dim_yx: Tuple[int, int]
    wavelength: float
    na: float
    pixel_size: float
    RI_medium: float
    padding_yx: Tuple[int, int]= (0, 0)
    mean_background_amp: float = 1.0
    wavelength_exc: float = 0.5


@dataclass
class SystemParameters3D:
    dim_zyx: Tuple[int, int, int]
    wavelength: float
    wavelength_exc: float
    na: float
    pixel_size: float
    pixel_size_z: float
    RI_medium: float
    padding_zyx: Tuple[int, int, int]


def apodization(img_param: Union[SystemParameters3D, SystemParameters],
                k0mag: float, norder: int = 3, inverted=False, min_height=1e-2):
    if hasattr(img_param, 'dim_zyx'):
        dim_zyx = (img_param.dim_zyx[0] + img_param.padding_zyx[0] * 2,
                   img_param.dim_zyx[1] + img_param.padding_zyx[1] * 2,
                   img_param.dim_zyx[2] + img_param.padding_zyx[2] * 2)
        pixel_size_z = img_param.pixel_size_z
    elif hasattr(img_param, 'dim_yx'):
        dim_zyx = (1, img_param.padding_yx[0] * 2 + img_param.dim_yx[0],
                   img_param.padding_yx[1] * 2 + img_param.dim_yx[1])
        pixel_size_z = 1
    else:
        raise ValueError("img_param should be either SystemParameters or SystemParameters3D")

    krdist_cutoff = img_param.na * 2 / img_param.wavelength
    kzdist_cutoff = [np.ceil((1 - np.cos(img_param.na / img_param.RI_medium)) / img_param.wavelength), ]
    kzdist_cutoff.append(kzdist_cutoff[0] * (1 + img_param.wavelength/img_param.wavelength_exc))
    kzdist_cutoff.append(kzdist_cutoff[0] * 1.3)

    apocutoff = krdist_cutoff + k0mag * (norder - 1) * 0.8
    if norder >= 2:
        zapocutoff = kzdist_cutoff[1]
    else:
        zapocutoff = kzdist_cutoff[0]

    kxlin = np.fft.fftfreq(dim_zyx[2], img_param.pixel_size)
    kylin = np.fft.fftfreq(dim_zyx[1], img_param.pixel_size)
    kzlin = np.fft.fftfreq(dim_zyx[0], pixel_size_z)

    ky, kx = np.meshgrid(kylin, kxlin, indexing='ij')
    kr = np.sqrt(ky**2 + kx**2)

    rho = np.sqrt((kr / apocutoff)[np.newaxis]**2 + (kzlin / zapocutoff)[:, np.newaxis, np.newaxis]**2)
    rho = np.minimum(rho, 1)
    apo = 1 - rho * (1 - min_height)

    if hasattr(img_param, 'dim_yx'):
        apo = np.squeeze(apo, axis=0)

    if inverted:
        apo = 1 / apo

    return apo


def notch_filter(img_param: SystemParameters3D, order: int, d: float, w: float, kz_offset: float = 0,
                 inverted=False):
    assert d < 1.0, "d should be smaller than 1.0"

    krdist_cutoff = img_param.na * 2 / img_param.wavelength
    kzdist_cutoff = np.ceil((1 - np.cos(img_param.na / img_param.RI_medium)) / img_param.wavelength)

    dim_zyx = (img_param.dim_zyx[0] + img_param.padding_zyx[0] * 2,
               img_param.dim_zyx[1] + img_param.padding_zyx[1] * 2,
               img_param.dim_zyx[2] + img_param.padding_zyx[2] * 2)

    kxlin = np.fft.fftfreq(dim_zyx[2], img_param.pixel_size)
    kylin = np.fft.fftfreq(dim_zyx[1], img_param.pixel_size)
    kzlin = np.fft.fftfreq(dim_zyx[0], img_param.pixel_size_z)

    ky, kx = np.meshgrid(kylin, kxlin, indexing='ij')
    kr = np.sqrt(ky**2 + kx**2)

    if order == 1:
        kzlin = np.minimum(np.abs(kzlin - kz_offset), np.abs(kzlin + kz_offset))
        notch = 1 - d * np.exp(-(kr[np.newaxis]**2/krdist_cutoff**2 + kzlin[:, np.newaxis, np.newaxis]**2 / kzdist_cutoff**2) / 2 / w**2)
    else:
        notch = 1 - d * np.exp(-(kr[np.newaxis]**2/krdist_cutoff**2 + kzlin[:, np.newaxis, np.newaxis]**2 / kzdist_cutoff**2) / 2 / w**2)

    if inverted:
        notch = 1 / notch

    return notch


def psf_gaussian_approx(dim_yx: Tuple[int, int],
                        pixel_size: float,
                        na: float,
                        wavelength: float,
                        paraxial: bool = True,
                        ri: float = 1.0) -> np.ndarray:
    """
    Generate Gaussian-approx PSF. Based on: https://opg.optica.org/ao/fulltext.cfm?uri=ao-46-10-1819&id=130945

    Args:
        dim_yx: y-x matrix dimensions of the output
        pixel_size: pixel size in micron
        na: numerical aperture
        wavelength: emission wavelength in micron
        paraxial: whether to use paraxial approximation
        ri: refractive index of the medium

    Returns:
        psf: Gaussian-approximated PSF in 2D
    """

    xlin = genGridNumpy(dim_yx[1], pixel_size, flag_shift=True).real
    ylin = genGridNumpy(dim_yx[0], pixel_size, flag_shift=True).real
    y, x = np.meshgrid(ylin, xlin, indexing='ij')
    r = np.sqrt(y**2 + x**2)

    if paraxial:
        # paraxial approximation
        sigma = 0.21 * wavelength / na
    else:
        # nooparaxial, for high NA
        k_em = 2 * np.pi / wavelength
        alpha = np.arcsin(na / ri)
        sigma = 1 / (ri *  k_em) / np.sqrt((4 - 7 * np.cos(alpha)**1.5 + 3 * np.cos(alpha)**3.5) /
                                           (7 - 7 * np.cos(alpha)**1.5))

    psf = np.exp(-r ** 2 / (2 * sigma ** 2))
    return psf


def apodize_edge(img, napodize=10):
    img_out = img.copy()

    # top and bottom
    diff = (img[..., -1, :] - img[..., 0, :]) / 2
    l = np.arange(napodize)
    fact = 1.0 - np.sin((l + 0.5) / napodize * np.pi * 0.5)
    img_out[..., :napodize, :] += diff[..., np.newaxis, :] * fact[:, np.newaxis]
    img_out[..., -napodize:, :] -= diff[..., np.newaxis, :] * fact[::-1, np.newaxis]

    # left and right
    diff = (img[..., :, -1] - img[..., :, 0]) / 2
    img_out[..., :, :napodize] += diff[..., :, np.newaxis] * fact[np.newaxis, :]
    img_out[..., :, -napodize:] -= diff[..., :, np.newaxis] * fact[np.newaxis, ::-1]

    return img_out


def OTF_3D_fluo(param: SystemParameters3D, rfft: bool = False):
    """
    Compute for the optical transfer function (OTF) for 3D fluorescence microscopy systems based on angular spectrum
    propagation. Note that the z direction of the returned matrix is in real domain.

    Args:
        param: optical parameter for 3D z-scan system
        rfft: whether to return OTF for real FFT

    Returns:
        OTF: optical transfer function in (z, fy, fx)
    """

    dim_zyx = (param.dim_zyx[0] + param.padding_zyx[0] * 2,
               param.dim_zyx[1] + param.padding_zyx[1] * 2,
               param.dim_zyx[2] + param.padding_zyx[2] * 2)

    zlin = np.asarray(genGridNumpy(dim_zyx[0], param.pixel_size_z, flag_shift=True))
    fxlin = np.asarray(genGridNumpy(dim_zyx[2], 1/param.pixel_size/dim_zyx[2], flag_shift=True))
    fylin = np.asarray(genGridNumpy(dim_zyx[1], 1/param.pixel_size/dim_zyx[1], flag_shift=True))

    fyy, fxx = np.meshgrid(fylin, fxlin, indexing='ij')

    r_obj = np.sqrt(fxx ** 2 + fyy ** 2).astype(np.complex128)
    pupil = np.zeros(dim_zyx[1:], dtype=np.complex128)
    pupil[r_obj.real < (param.na/param.wavelength)] = 1.0
    H = np.exp(1.0j * 2 * np.pi / param.wavelength * zlin.astype(np.complex128)[:, np.newaxis, np.newaxis] *
               np.real(np.sqrt(param.RI_medium - param.wavelength**2 * r_obj[np.newaxis, :, :]**2)) *
               pupil[np.newaxis, :, :]) * pupil[np.newaxis, :, :]
    psf = np.abs(np.fft.ifft2(H, axes=(1, 2)))**2

    if rfft:
        OTF = np.fft.rfftn(psf, axes=(0, 1, 2))
    else:
        OTF = np.fft.fftn(psf, axes=(0, 1, 2))

    return OTF, psf


def generate_linear_motion(t, start_pos_yx, end_pos_yx, rot_start=0, rot_end=0):
    rot = (1 - t) * rot_start + t * rot_end
    y = (1 - t) * start_pos_yx[0] + t * end_pos_yx[0]
    x = (1 - t) * start_pos_yx[1] + t * end_pos_yx[1]
    return np.array([x, y, rot]).transpose()


def generate_affine_motion(t, start_pos_yx, end_pos_yx, rot_start=0, rot_end=0, scale_start=1, scale_end=1,
                           shear_start=0, shear_end=0):
    rot = (1 - t) * rot_start + t * rot_end
    scale = (1 - t) * scale_start + t * scale_end
    shear = (1 - t) * shear_start + t * shear_end
    y = (1 - t) * start_pos_yx[0] + t * end_pos_yx[0]
    x = (1 - t) * start_pos_yx[1] + t * end_pos_yx[1]
    return np.array([x, y, rot, scale, shear]).transpose()


def object_transform(obj: np.ndarray, target_dim_yx: Tuple[int, int],
                     coord: Union[List[float], Tuple[float, float, float, float]]) -> np.ndarray:
    """
    Linear transformation of a given object.

    :param obj: original object to be transformed
    :param target_dim_yx: output matrix dimension
    :param coord: tuple or list to specify the transformation as (x, y, orientation, scale)
    :return obj_transformed: transformed object
    """
    scaling_factor = min(target_dim_yx[0] / obj.shape[0], target_dim_yx[1] / obj.shape[1])

    obj = transform.resize(obj, (int(obj.shape[0] * scaling_factor), int(obj.shape[1] * scaling_factor)),
                           anti_aliasing=True, order=1)
    obj = np.pad(obj, (((target_dim_yx[0] - obj.shape[0]) // 2,
                        target_dim_yx[0] - (target_dim_yx[0] - obj.shape[0]) // 2 - obj.shape[0]),
                       ((target_dim_yx[1] - obj.shape[1]) // 2,
                        target_dim_yx[1] - (target_dim_yx[1] - obj.shape[1]) // 2 - obj.shape[1])))

    # shrink, rotate, translate
    trans_shrink = transform.SimilarityTransform(scale=coord[3])
    trans_shift1 = transform.SimilarityTransform(translation=[target_dim_yx[1] * (1 - coord[3]) * 0.5,
                                                              target_dim_yx[0] * (1 - coord[3]) * 0.5])
    trans_rotate = transform.SimilarityTransform(rotation=np.deg2rad(coord[2]))
    trans_shift2 = transform.SimilarityTransform(translation=[-target_dim_yx[1] * 0.5, -target_dim_yx[0] * 0.5])
    trans_shift2_inv = transform.SimilarityTransform(translation=[target_dim_yx[1] * 0.5, target_dim_yx[0] * 0.5])
    trans_shift3 = transform.SimilarityTransform(translation=[coord[0], coord[1]])

    obj_transformed = transform.warp(obj, (trans_shrink + trans_shift1 + trans_shift2 + trans_rotate +
                                           trans_shift2_inv + trans_shift3).inverse, order=1)
    return obj_transformed


def object_transform_affine(obj: np.ndarray, target_dim_yx: Tuple[int, int],
                            coord: Union[List[float], Tuple[float, float, float, float, float]]) -> np.ndarray:
    """
    Affine transformation of a given object.

    :param obj: original object to be transformed
    :param target_dim_yx: output matrix dimension
    :param coord: tuple or list to specify the transformation as (x, y, orientation, scale, shear)
    :return obj_transformed: transformed object
    """
    scaling_factor = min(target_dim_yx[0] / obj.shape[0], target_dim_yx[1] / obj.shape[1])

    obj = transform.resize(obj, (int(obj.shape[0] * scaling_factor), int(obj.shape[1] * scaling_factor)),
                           anti_aliasing=True)
    obj = np.pad(obj, (((target_dim_yx[0] - obj.shape[0]) // 2,
                        target_dim_yx[0] - (target_dim_yx[0] - obj.shape[0]) // 2 - obj.shape[0]),
                       ((target_dim_yx[1] - obj.shape[1]) // 2,
                        target_dim_yx[1] - (target_dim_yx[1] - obj.shape[1]) // 2 - obj.shape[1])))

    # shrink, rotate, translate
    trans = transform.AffineTransform(scale=1/coord[3], rotation=np.deg2rad(coord[2]), shear=np.deg2rad(coord[4]),
                                      translation=[coord[0]/coord[3], coord[1]/coord[3]])
    trans_shift1 = transform.SimilarityTransform(translation=[target_dim_yx[1] * (1 - 1/coord[3]) * 0.5,
                                                              target_dim_yx[0] * (1 - 1/coord[3]) * 0.5])
    obj_transformed = transform.warp(obj, trans + trans_shift1)
    return obj_transformed


def object_transform_swirl(obj: np.ndarray, target_dim_yx: Tuple[int, int], scale, strength, radius):

    scaling_factor = min(target_dim_yx[0] / obj.shape[0], target_dim_yx[1] / obj.shape[1])

    obj = transform.resize(obj, (int(obj.shape[0] * scaling_factor), int(obj.shape[1] * scaling_factor)),
                           anti_aliasing=True)
    obj = np.pad(obj, (((target_dim_yx[0] - obj.shape[0]) // 2,
                        target_dim_yx[0] - (target_dim_yx[0] - obj.shape[0]) // 2 - obj.shape[0]),
                       ((target_dim_yx[1] - obj.shape[1]) // 2,
                        target_dim_yx[1] - (target_dim_yx[1] - obj.shape[1]) // 2 - obj.shape[1])))
    trans_shrink = transform.SimilarityTransform(scale=scale)
    trans_shift1 = transform.SimilarityTransform(translation=[target_dim_yx[1] * (1 - scale) * 0.5,
                                                              target_dim_yx[0] * (1 - scale) * 0.5])
    obj = transform.warp(obj, (trans_shrink + trans_shift1).inverse)

    obj_transformed = transform.swirl(obj, rotation=0, strength=strength, radius=radius)
    return obj_transformed


def brownian(x0, n, dt, delta, seed, out=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    seed : int or generator instance
        Random seed for reproducibility.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.

    Source: https://scipy-cookbook.readthedocs.io/items/BrownianMotion.html
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(size=x0.shape + (n,), scale=delta * np.sqrt(dt), random_state=seed)

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def generate_rod_phantom(dim_yx: Tuple[int, int], num_rods: int, rod_length: int, rod_width: int,
                         triangle_mix: bool = False, seed: int = 219) -> np.ndarray:
    """Draw a phantom with rods of length `rod_length` and width `rod_width` on a matrix of size `dim_yx`.

    Args:
        dim_yx: tuple of integers, size of the output matrix
        num_rods: number of rods to draw
        rod_length: length of the rod
        rod_width: width of the rod
        triangle_mix: whether to draw half as triangle rods
        seed: random seed for reproducibility

    Returns:
        obj: phantom matrix with rods
    """
    rng = np.random.default_rng(seed)
    obj = np.zeros(dim_yx)
    for _ in range(num_rods):
        line = np.zeros_like(obj)
        x = rng.integers((rod_length+1) // 2, dim_yx[1] - (rod_length+1)//2)
        y = rng.integers((rod_length+1) // 2, dim_yx[0] - (rod_length+1)//2)
        orientation = rng.uniform(0, np.pi * 2)
        x1 = x + rod_length * np.cos(orientation) * 0.5
        y1 = y + rod_length * np.sin(orientation) * 0.5
        x2 = x - rod_length * np.cos(orientation) * 0.5
        y2 = y - rod_length * np.sin(orientation) * 0.5

        triangle = rng.choice([True, False], p=[0.5, 0.5]) if triangle_mix else False
        if triangle:
            x21 = x2 + (2 * rod_width - 1) * np.cos(orientation + np.pi/2)
            y21 = y2 + (2 * rod_width - 1) * np.sin(orientation + np.pi/2)
            x22 = x2 - (2 * rod_width - 1) * np.cos(orientation + np.pi/2)
            y22 = y2 - (2 * rod_width - 1) * np.sin(orientation + np.pi/2)
            rr, cc = skimage.draw.polygon([y1, y21, y22], [x1, x21, x22], shape=dim_yx)
        else:
            rr, cc = skimage.draw.line(int(y1), int(x1), int(y2), int(x2))
        line[rr, cc] = 1.0
        if not triangle and rod_width > 1:
            line = skimage.morphology.dilation(line, skimage.morphology.disk(rod_width - 1))
        obj += line
    return np.minimum(obj, 1.0)


class PhantomTemporal:
    def __init__(self, param):
        self.param = param
        self.xlin          = genGridNumpy(self.param.dim_yx[1], self.param.pixel_size)
        self.ylin          = genGridNumpy(self.param.dim_yx[0], self.param.pixel_size)

    def generate_bead_phantom(self, coordinates, phase=1.0):
        # coordinates: [(x, y, orientation, scale)] orientation doesn't matter for now

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.complex128)
        obj_fluo = np.zeros(self.param.dim_yx)
        for coord in coordinates:
            additive_phase = (np.maximum(coord[3]**2 - (self.ylin[:, np.newaxis] - coord[1])**2 -
                                         (self.xlin[np.newaxis, :] - coord[0])**2, 0.0))**0.5 * 2 * phase
            obj_phase += additive_phase

            additive_fluo = np.abs(np.maximum((1.5 * self.param.pixel_size)**2 - (self.ylin[:, np.newaxis] - coord[1])**2 -
                                         (self.xlin[np.newaxis, :] - coord[0])**2, 0.0))**0.5
            obj_fluo += additive_fluo
        return obj_phase, obj_fluo

    def generate_shepp_logan(self, coordinates, max_value=1.0, max_phantom=1.0):
        # coordinates: [(x, y, orientation, scale), ...]

        obj = np.zeros(self.param.dim_yx)

        for coord in coordinates:
            phantom = np.minimum(data.shepp_logan_phantom(), max_phantom)
            phantom_transformed = object_transform(
                phantom, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3]))

            obj += phantom_transformed

        obj = obj / (np.max(obj) + 1e-8) * max_value

        return obj

    def generate_shepp_logan_2channel(self, coordinates):
        # coordinates: [(x, y, orientation, scale), ...]

        obj = np.zeros((2, ) + self.param.dim_yx)

        for coord in coordinates:
            phantom = data.shepp_logan_phantom()
            phantom_1 = (phantom >= 0.7).astype(np.float) * 0.2
            phantom[phantom >= 0.7] = 0
            phantom_transformed = object_transform(
                phantom, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3]))
            phantom_1_transformed = object_transform(
                phantom_1, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3]))

            obj[0] += phantom_transformed
            obj[1] += phantom_1_transformed

        return obj.transpose((1, 2, 0))

    def generate_shepp_logan_affine(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, orientation, scale, shear), ...]

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.complex128)
        obj_fluo = np.zeros(self.param.dim_yx)

        for coord in coordinates:
            phantom = data.shepp_logan_phantom()
            phantom_fluo = (phantom >= 0.7).astype(np.float)
            phantom[phantom >= 0.7] = 0
            phantom_transformed = object_transform_affine(
                phantom, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3], coord[4]))
            phantom_fluo_transformed = object_transform_affine(
                phantom_fluo, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3], coord[4]))

            phantom_transformed = phantom_transformed / np.max(phantom_transformed) * max_value
            obj_phase +=  phantom_transformed
            obj_fluo += phantom_fluo_transformed

        return obj_phase, obj_fluo

    def generate_shepp_logan_swirl(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, scale, strengh, radius), ...]
        # x, y are dummy var here
        obj_phase = np.zeros(self.param.dim_yx, dtype=np.complex128)
        obj_fluo = np.zeros(self.param.dim_yx)

        for coord in coordinates:
            phantom = data.shepp_logan_phantom()
            phantom_fluo = (phantom >= 0.7).astype(np.float)
            phantom[phantom >= 0.7] = 0
            phantom_transformed = object_transform_swirl(
                phantom, self.param.dim_yx, coord[2], coord[3], coord[4])
            phantom_fluo_transformed = object_transform_swirl(
                phantom_fluo, self.param.dim_yx, coord[2], coord[3], coord[4])

            phantom_transformed = phantom_transformed / np.max(phantom_transformed) * max_value
            obj_phase +=  phantom_transformed
            obj_fluo += phantom_fluo_transformed

        return obj_phase, obj_fluo

    def generate_usaf_target(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, orientation, scale), ...]
        filepath = 'assets/USAF-1951.png'
        img_dim_hw = [1550, 1550]
        im_frame = Image.open(filepath).convert('L')
        phantom = 255 - np.asarray(im_frame)
        phantom = transform.resize(phantom, (400, 400), anti_aliasing=True)

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.float64)
        for coord in coordinates:
            phantom_transformed = object_transform(
                phantom, self.param.dim_yx,
                (coord[0]/self.param.pixel_size, coord[1]/self.param.pixel_size, coord[2], coord[3]))

            phantom_transformed = phantom_transformed.astype(np.float64) / np.max(phantom_transformed) * max_value
            obj_phase += phantom_transformed
        return obj_phase

    def generate_usaf_target_affine(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, orientation, scale, shear), ...]
        filepath = 'assets/USAF-1951.png'
        img_dim_hw = [1550, 1550]
        im_frame = Image.open(filepath).convert('L')
        phantom = 255 - np.asarray(im_frame)
        phantom = transform.resize(phantom, (400, 400), anti_aliasing=True)

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.float64)
        for coord in coordinates:
            phantom_transformed = object_transform_affine(
                phantom, self.param.dim_yx,
                (coord[0] / self.param.pixel_size, coord[1] / self.param.pixel_size, coord[2], coord[3], coord[4]))

            phantom_transformed = phantom_transformed.astype(np.float64) / np.max(phantom_transformed) * max_value
            obj_phase += phantom_transformed
        return obj_phase

    def generate_usaf_target_shear(self,
                                   shear: float,
                                   scale: float = 0.5,
                                   max_value=1.0):
        # this function handles shear than affine transformation. the center will stay the same after shear.
        filepath = 'assets/USAF-1951.png'
        img_dim_hw = [1550, 1550]
        im_frame = Image.open(filepath).convert('L')
        phantom = 255 - np.asarray(im_frame)
        phantom = transform.resize(phantom, self.param.dim_yx, anti_aliasing=True).astype(np.float64)

        trans_shrink = transform.SimilarityTransform(scale=scale)
        trans_shear = transform.AffineTransform(np.array([[1, shear, 0], [0, 1, 0], [0, 0, 1]]))
        trans_shift = transform.SimilarityTransform(translation=[-shear * phantom.shape[1] * 0.5, 0])
        trans_shift2 = transform.SimilarityTransform(translation=[phantom.shape[1] * 0.5 * (1 - scale),
                                                                  phantom.shape[0] * 0.5 * (1 - scale)])

        obj = transform.warp(phantom, (trans_shear + trans_shift + trans_shrink + trans_shift2).inverse, order=1, mode='constant', cval=0, preserve_range=True)
        obj = obj.astype(np.float64) / np.max(obj) * max_value
        return obj

    def generate_usaf_target_swirl(self, coordinates, max_value=1.0):
        # coordinates: [(x, y, scale, strengh, radius), ...]
        # x, y are dummy var here
        filepath = 'assets/USAF-1951.png'
        img_dim_hw = [1550, 1550]
        im_frame = Image.open(filepath).convert('L')
        phantom = 255 - np.asarray(im_frame)
        phantom = transform.resize(phantom, (400, 400), anti_aliasing=True)

        obj_phase = np.zeros(self.param.dim_yx, dtype=np.float64)
        for coord in coordinates:
            phantom_transformed = object_transform_swirl(
                phantom, self.param.dim_yx, coord[2], coord[3], coord[4])

            phantom_transformed = phantom_transformed.astype(np.float64) / np.max(phantom_transformed) * max_value
            obj_phase += phantom_transformed
        return obj_phase


def load_video(filename, fov=None, single_channel=False, target_dim=None):
    vid = imageio.get_reader(filename, 'ffmpeg')
    ret = []
    for image in vid.iter_data():
        if single_channel:
            image = np.mean(image, axis=-1)
        if fov:
            image = image[fov[0]:fov[1], fov[2]:fov[3]]
        if target_dim:
            image = transform.resize(image, target_dim)
        ret.append(image)

    return np.array(ret)


def hotpixel_removal(imgs, int_thres_percentile=85, pixel_on_rate=0.9, detect_only=False):
    num_img = imgs.shape[0]
    thres = np.percentile(imgs, int_thres_percentile)
    count = np.sum(imgs > thres, axis=0)
    hotpixel_mask = count > (num_img * pixel_on_rate)

    print('Thresholding at pixel intensity value {} and pixel on rate {}.'.format(thres, pixel_on_rate))

    if detect_only:
        return hotpixel_mask

    imgs_median = np.array([skimage.filters.median(img, skimage.morphology.disk(2)) for img in imgs])
    out = imgs * (1 - hotpixel_mask)[np.newaxis] + imgs_median * hotpixel_mask[np.newaxis]

    return out, hotpixel_mask
