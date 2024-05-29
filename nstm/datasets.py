# -*- coding: utf-8 -*-
"""Data utilities and dataset loader for 3D SIM data.

The data loader is designed to load 3D SIM data with OTF and timestamp information. The data loader support loading .tif
and .czi file (from Zeiss) format.
"""

import dataclasses
import glob
import os
from typing import Tuple, Union, List
import numpy as np
from aicsimageio import AICSImage
from skimage import transform
from skimage import io as skimageio
import sqlite3
import pandas as pd
import scipy.signal
from aicsimageio.readers.czi_reader import CziReader


class Datasets:
    def __init__(self, dir_path: str):
        self.dir_path = dir_path
        self.image = None

    def load(self, data_name, ndim=3, fov_yxhw=None, resize_hw=None):
        format_suffix = '*.tif'
        filepath = sorted(glob.glob(self.dir_path + '/' + data_name + '/' + format_suffix))
        if len(filepath) == 0:
            raise FileNotFoundError
        elif len(filepath) == 1:
            img = AICSImage(filepath).data
        else:
            img = np.squeeze(np.array([AICSImage(f).data for f in filepath]))

        if fov_yxhw:
            img = img[..., fov_yxhw[0]:fov_yxhw[0]+fov_yxhw[2], fov_yxhw[1]:fov_yxhw[1]+fov_yxhw[3]]

        if resize_hw:
            if img.ndim == 2:
                img = transform.resize(img, resize_hw)
            elif img.ndim == 3:
                img = np.array([transform.resize(im, resize_hw) for im in img])
            elif img.ndim == 4:
                img = np.array([[transform.resize(i, resize_hw) for i in im] for im in img])

        return img

    def list(self):
        return os.walk(self.dir_path)[1]


def image_upsampling(I_image, upsamp_factor=1.0, bg=0):
    F = lambda x: np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(x)))
    iF = lambda x: np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(x)))

    Nimg, Ncrop, Mcrop = I_image.shape
    if upsamp_factor == 1:
        return I_image

    N = int(Ncrop * upsamp_factor)
    M = int(Mcrop * upsamp_factor)

    I_image_up = np.zeros((Nimg, N, M))

    for i in range(0, Nimg):
        I_image_up[i] = abs(iF(np.pad(F(np.maximum(0, I_image[i] - bg)),
                                      (((N - Ncrop) // 2,), ((M - Mcrop) // 2,)), mode='constant')))

    return I_image_up


def image_resizing(I_image, out_dim):
    out_images = np.zeros((I_image.shape[0], out_dim[0], out_dim[1]))
    for i in range(I_image.shape[0]):
        out_images[i] = transform.resize(I_image[i], out_dim)
    return out_images


@dataclasses.dataclass
class SIM3DDataLoader:
    otf_path: str
    meta_path: str
    zoomfact: float
    ndirs: int
    nphases: int
    start_timepoint: int

    def load_3DSIM_raw(self, raw_path, fov_zyxshw, noise_std=0, background_int=0, normalize=True, i_timepoint_czi=None):
        if raw_path.lower().endswith('.tif') or raw_path.lower().endswith('.tiff'):
            print("Loading tiff file: {}".format(raw_path))
            img_raw = skimageio.imread(raw_path)
        elif raw_path.lower().endswith('.czi'):
            print("Loading czi file: {}".format(raw_path))
            img_handler = CziReader(raw_path)
            img_raw = img_handler.data
            print('CZI file shape: {}'.format(img_raw.shape))
            if i_timepoint_czi is None:
                i_timepoint_czi = self.start_timepoint
            img_raw = np.squeeze(img_raw, axis=(0, 4))[:, :, i_timepoint_czi]
            img_raw = img_raw.transpose((2, 1, 0, 3, 4))  # (Z, ndirs, nphases, H, W)
            img_raw = img_raw.reshape((-1, img_raw.shape[3], img_raw.shape[4]))
        else:
            raise NotImplementedError('{} not supported. Only support .tif and .czi file format'.format(raw_path))

        if fov_zyxshw is None:
            dim_z = img_raw.shape[0] // self.ndirs // self.nphases
            fov_zyxshw = (0, 0, 0, dim_z, img_raw.shape[-2], img_raw.shape[-1])

        img = np.maximum(img_raw[:, fov_zyxshw[1]:fov_zyxshw[1] + fov_zyxshw[4],
          fov_zyxshw[2]:fov_zyxshw[2] + fov_zyxshw[5]].astype(np.float64) - background_int, 0)

        if normalize:
            img = img / np.max(img)

        img = img.reshape((-1, self.ndirs, self.nphases, fov_zyxshw[4], fov_zyxshw[5])).transpose((1, 2, 0, 3, 4))
        img = img[:, :, fov_zyxshw[0]:fov_zyxshw[0] + fov_zyxshw[3]]

        if noise_std > 0:
            import warnings
            warnings.warn('noise_std arg is deprecated. No denoising is performed.', DeprecationWarning)

        img = np.array([[image_upsampling(img__, self.zoomfact) for img__ in img_] for img_ in img])

        return img

    def load_3DSIM_OTF(self, list_otf_path=None):
        if list_otf_path is None:
            OTF = [self.load_OTF(self.otf_path) for _ in range(self.ndirs)]
        elif len(list_otf_path) == 1:
            OTF = [self.load_OTF(list_otf_path[0]) for _ in range(self.ndirs)]
        else:
            OTF = [self.load_OTF(otf_path) for otf_path in list_otf_path]
        return OTF

    @staticmethod
    def load_OTF(otf_path):
        OTF_raw = skimageio.imread(otf_path)
        OTF = OTF_raw[:, ::2, :] + OTF_raw[:, 1::2, :] * 1.0j
        return OTF

    def load_metadata(self, normalize=True, avg_phase=True, single_plane_time=False):
        if self.meta_path.lower().endswith('.sqlite3'):
            print("Loading metadata from sqlite3 file: {}".format(self.meta_path))
            conn = sqlite3.connect(self.meta_path, uri=True)
            df = pd.read_sql_query("SELECT * FROM SliceList", conn)

            if hasattr(self, 'num_timepoint'):
                filtered_df = df[(df['ImageList_id'] >= self.start_timepoint + 1) &
                                 (df['ImageList_id'] <= self.start_timepoint + self.num_timepoint)].sort_values('SliceList_id')
            else:
                filtered_df = df[df['ImageList_id'] == self.start_timepoint+1].sort_values('SliceList_id')

            time = filtered_df['Encoder_Timestamp_ms'].to_numpy() * 1e-3
            time = time.reshape((-1, self.ndirs, self.nphases)).transpose((1, 2, 0))

            z_loc = df['Z_Target_Position_um'].to_numpy()
            z_loc = z_loc - np.min(z_loc)
            z_loc = z_loc.reshape((-1, self.ndirs, self.nphases)).transpose((1, 2, 0))
        elif self.meta_path.lower().endswith('.csv'):
            print("Loading metadata from csv file: {}".format(self.meta_path))
            df = pd.read_csv(self.meta_path)
            time = df['timestamp'].to_numpy() * 1e-3
            z_loc = df['z'].to_numpy()

            assert len(np.unique(df['phase'])) == self.nphases, 'number of phases in metadata does not match with nphases'
            assert len(np.unique(df['rot'])) == self.ndirs, 'number of rotations in metadata does not match with ndirs'
            nz = len(np.unique(df['z'].to_numpy()))
            ntimepoints = len(np.unique(df['timepoint'].to_numpy()))

            time = time.reshape((ntimepoints, self.ndirs, nz, self.nphases))
            z_loc = z_loc.reshape((ntimepoints, self.ndirs, nz, self.nphases))

            if hasattr(self, 'num_timepoint'):
                num_timepoint = self.num_timepoint
            else:
                num_timepoint = 1

            time = time[self.start_timepoint:self.start_timepoint + num_timepoint].transpose((1, 3, 0, 2)).reshape((self.ndirs, self.nphases, -1))
            z_loc = z_loc[self.start_timepoint:self.start_timepoint + num_timepoint].transpose((1, 3, 0, 2)).reshape((self.ndirs, self.nphases, -1))
        else:
            raise NotImplementedError('{} not supported. Only support .sqlite3 and .csv file format'.format(self.meta_path))

        if avg_phase:
            time = np.mean(time, axis=1)
            z_loc = np.mean(z_loc, axis=1)

        if single_plane_time:
            time = time[:, :, 0]

        time = time - np.min(time)

        if normalize:
            time = time / np.max(time) * 2 - 1
            z_loc = z_loc / np.maximum(np.max(z_loc), 1e-6) * 2 - 1

        return time, z_loc


@dataclasses.dataclass
class SIM3DDataLoaderMultitime(SIM3DDataLoader):
    num_timepoint: int

    def list_files(self, raw_path_regex, ):
        list_raw_path = sorted(glob.glob(raw_path_regex))
        assert len(list_raw_path) >= 1, 'No image files found in {}'.format(raw_path_regex)

        return list_raw_path[self.start_timepoint:self.start_timepoint+self.num_timepoint]

    def load_3DSIM_raw(self, raw_path_regex, fov_zyxshw, noise_std=0, background_int=0, normalize=True):
        if raw_path_regex.lower().endswith('.czi'):
            list_raw_path = [raw_path_regex for _ in range(self.num_timepoint)]
        else:
            list_raw_path = self.list_files(raw_path_regex)
            assert len(list_raw_path) == self.num_timepoint, 'number of timepoints does not match with num_timepoint'

        list_img = np.zeros((self.ndirs, self.nphases, fov_zyxshw[3] * self.num_timepoint,
                             int(fov_zyxshw[4]*self.zoomfact), int(fov_zyxshw[5]*self.zoomfact)))
        for i in range(self.num_timepoint):
            list_img[:, :, fov_zyxshw[3]*i:fov_zyxshw[3]*(i+1)] = super().load_3DSIM_raw(
                raw_path=list_raw_path[i], fov_zyxshw=fov_zyxshw, noise_std=noise_std,
                background_int=background_int, normalize=False, i_timepoint_czi=i+self.start_timepoint)

        if normalize:
            list_img = list_img / np.max(list_img)

        return list_img
