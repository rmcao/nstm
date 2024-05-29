# -*- coding: utf-8 -*-
"""Utility functions for differential phase contrast (DPC) imaging.

This module contains the utility functions for differential phase contrast (DPC) imaging. The functions include loading
illumination patterns, generating source patterns, generating transfer functions, and conventional Tikhonov solver.

This script heavily references from these two repositories:
https://github.com/Waller-Lab/DPC
https://github.com/Waller-Lab/DPC_withAberrationCorrection
"""

import numpy as np
from skimage import io as skimageio
import skimage.transform
import skimage
from scipy.signal import fftconvolve

from nstm import utils


def load_illum_pattern(param: utils.SystemParameters, meta, list_led_na, large_led=True, first_n=None):
    fxlin = np.fft.fftfreq(param.dim_yx[1], param.pixel_size)
    fylin = np.fft.fftfreq(param.dim_yx[0], param.pixel_size)
    fy, fx = np.meshgrid(fylin, fxlin, indexing='ij')

    if first_n is None:
        first_n = len(meta['frame_state_list'])

    source = np.zeros((first_n, param.dim_yx[0], param.dim_yx[1], 3))

    for i in range(first_n):

        for d in meta['frame_state_list'][i]['illumination']['sequence'][0]:
            for i_c, c in enumerate(['r', 'g', 'b']):
                s = source[i, :, :, i_c]

                if d['value'][c] == 0:
                    continue

                v = d['value'][c] / 65535.0

                fxfy = (list_led_na[d['index']][0] / param.wavelength, list_led_na[d['index']][1] / param.wavelength)
                if large_led:
                    s[(fylin[:, np.newaxis] - fxfy[1]) ** 2 + (fxlin[np.newaxis, :] - fxfy[0]) ** 2 < 8e-3] = v # 8e-4
                else:
                    s[np.unravel_index(np.argmin(
                        (fylin[:, np.newaxis] - fxfy[1]) ** 2 + (fxlin[np.newaxis, :] - fxfy[0]) ** 2), s.shape)] = v
                source[i, :, :, i_c] = s

    return source


def genSourceAngular(sourceCoeffs, rotationAngle, imgSize, systemNa, ps, wavelength):
    dTheta = 360 / np.size(sourceCoeffs, 0)
    angleList = np.arange(0, 360, dTheta) + rotationAngle

    # Spatial coordinates
    M = imgSize[0]
    N = imgSize[1]

    # Spatial frequency coordinates
    dfx = 1 / (N * ps)
    dfy = 1 / (M * ps)
    fx = dfx * np.arange(-(N - np.mod(N, 2)) / 2, (N - np.mod(N, 2)) / 2 - (np.mod(N, 2) == 1))
    fy = dfy * np.arange(-(M - np.mod(M, 2)) / 2, (M - np.mod(M, 2)) / 2 - (np.mod(M, 2) == 1))
    [fxx, fyy] = np.meshgrid(fx, fy, indexing='ij')

    # SourceList Definition
    sourceList = np.zeros((len(angleList), M, N))
    for aIdx in range(0, len(angleList)):
        M1 = (np.sqrt(fxx ** 2 + fyy ** 2) < (systemNa / wavelength))
        M2 = (-np.sin(np.deg2rad(angleList[aIdx] + 180 + dTheta)) * fxx) >= (np.cos(np.deg2rad(angleList[aIdx] + dTheta + 180)) * fyy)
        M3 = (-np.sin(np.deg2rad(angleList[aIdx] + 180)) * fxx < np.cos(np.deg2rad(angleList[aIdx] + 180)) * fyy)
        sourceList[aIdx, :, :] = (M1 * M2 * M3) * sourceCoeffs[aIdx]

    return sourceList


def sourceGen(dim_yx, na, ps, wavelength, rotation=None):
    '''
    Generate DPC source patterns based on the rotation angles and numerical aperture of the illuminations.
    '''
    if rotation is None:
        rotation = [0, 90, 180, 270]
    source = []
    M = dim_yx[0]
    N = dim_yx[1]

    dfx = 1 / (N * ps)
    dfy = 1 / (M * ps)
    fx = dfx * np.arange(-(N - np.mod(N, 2)) / 2, (N - np.mod(N, 2)) / 2 - (np.mod(N, 2) == 1))
    fy = dfy * np.arange(-(M - np.mod(M, 2)) / 2, (M - np.mod(M, 2)) / 2 - (np.mod(M, 2) == 1))
    pupil = np.array(fx[np.newaxis, :] ** 2 + fy[:, np.newaxis] ** 2 <= (na / wavelength) ** 2, dtype="float32")

    for rot_index in range(len(rotation)):
        source.append(np.zeros((dim_yx), dtype='float32'))
        rotdegree = rotation[rot_index]
        if rotdegree < 180:
            source[-1][fy[:, np.newaxis] * np.cos(np.deg2rad(rotdegree)) + 1e-15 >=
                       fx[np.newaxis, :] * np.sin(np.deg2rad(rotdegree))] = 1.0
            source[-1] *= pupil
        else:
            source[-1][fy[:, np.newaxis] * np.cos(np.deg2rad(rotdegree)) + 1e-15 <
                       fx[np.newaxis, :] * np.sin(np.deg2rad(rotdegree))] = -1.0
            source[-1] *= pupil
            source[-1] += pupil
    source = np.asarray(source)
    return source


def gen_transfer_func(list_source: np.ndarray, pupil: np.ndarray, wavelength: float, shifted_out=True):
    DC = np.sum((np.abs(pupil[np.newaxis])**2 * list_source), axis=(-2, -1))

    M = np.fft.fft2(list_source * pupil[np.newaxis], axes=(-2, -1)) * np.conj(np.fft.fft2(pupil))

    Hu = 2 * np.fft.ifft2(M.real, axes=(-2, -1)) / DC[:, np.newaxis, np.newaxis]
    Hp = 1j * 2 * np.fft.ifft2(1j * M.imag, axes=(-2, -1)) / DC[:, np.newaxis, np.newaxis] / wavelength

    if shifted_out:
        Hu = np.fft.fftshift(Hu, axes=(-2, -1))
        Hp = np.fft.fftshift(Hp, axes=(-2, -1))

    return Hu, Hp


def genMeasurmentsLinear(complexField, Hu, Hp):
    # Generate Measurment (Linear)
    out= np.abs(np.fft.ifft2(((Hu * np.fft.fft2(np.abs(complexField))) + (Hp * np.fft.fft2(np.angle(complexField))))))

    # Background subtration
    out = out - np.mean(out)

    return out


def dpc_tikhonov_solver(imgs, Hu, Hp, amp_reg=5e-5, phase_reg=5e-5, wavelength=0.515):

    AHA = [(Hu.conj() * Hu).sum(axis=0) + amp_reg, (Hu.conj() * Hp).sum(axis=0),
           (Hp.conj() * Hu).sum(axis=0), (Hp.conj() * Hp).sum(axis=0) + phase_reg]
    determinant = AHA[0] * AHA[3] - AHA[1] * AHA[2]
    fIntensity = np.fft.fft2(imgs, axes=(-2, -1))
    AHy = np.asarray([(Hu.conj() * fIntensity).sum(axis=0), (Hp.conj() * fIntensity).sum(axis=0)])
    absorption = np.fft.ifft2((AHA[3] * AHy[0] - AHA[1] * AHy[1]) / determinant, axes=(-2, -1)).real
    phase = np.fft.ifft2((AHA[0] * AHy[1] - AHA[2] * AHy[0]) / determinant, axes=(-2, -1)).real

    return absorption, phase
