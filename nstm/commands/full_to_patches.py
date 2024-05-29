# -*- coding: utf-8 -*-
"""Cut a large image into patches for patch-based 3D SIM reconstruction and later merge them together.

Todo:
    * Clean up the code and remove unused functions.
"""

import json
import os
from os import path
from absl import flags
from absl import app
import numpy as np
from scipy.signal import convolve2d
import tifffile

FLAGS = flags.FLAGS


def define_flags():
    flags.DEFINE_bool("merge", False, "combine all patches into a single image.")
    flags.DEFINE_bool("write_tiff", False, "output a single tiff image stack (combining all timepoints).")
    flags.DEFINE_string("run_name", "", "(deprecated) The name (prefix) of the experiment.")
    flags.DEFINE_string("save_dir", None, "The directory to save the output.")
    flags.DEFINE_enum("overlap_mode", "mid", ["avg", "max", "min", "mid", "grad"],
                      "The strategy to merge overlapped regions between patches.")
    flags.DEFINE_list("full_fov", None, "The field of view of the image, on each dimension, in pixels.")
    flags.DEFINE_list("target_fov", None,
                      "The field of view of the target patch, on each dimension, in pixels.")
    flags.DEFINE_integer("overlap", 10, "The overlap between two patches in pixels.")
    flags.DEFINE_list(
        "fov_offset", None,
        "The coord offset for the full fov. Used if only a part of the image is used.")
    flags.DEFINE_float("zoomfact", 2, "zoom factor on lateral dimensions.")


def cut_patches(full_dim, target_dim, overlap):
    assert (len(full_dim) == len(target_dim))

    if not isinstance(overlap, list) and not isinstance(overlap, tuple):
        overlap = (overlap,) * len(full_dim)

    num_patch, patch_coord = [], []
    for i in range(len(full_dim)):
        if full_dim[i] == target_dim[i] or full_dim[i] <= overlap[i]:
            num_patch.append(1)
        else:
            num_patch.append(np.ceil((full_dim[i] - overlap[i]) / np.maximum(target_dim[i] - overlap[i], 1)))
        coord_start = np.arange(num_patch[i]) * (target_dim[i] - overlap[i])
        coord_start[-1] = full_dim[i] - target_dim[i]
        patch_coord.append(coord_start)

    patch_coord_start = np.array(np.meshgrid(*patch_coord, indexing='ij')).reshape((len(full_dim),-1)).transpose()

    return patch_coord_start


def get_no_overlap_mask_2d(coord_start, patch_dim, full_dim, overlap, dtype=np.float32):
    mask = np.zeros((full_dim[0], full_dim[1]), dtype=dtype)

    if coord_start[0] == 0:
        start_ind = [0, -1]
    else:
        start_ind = [coord_start[0] + overlap, -1]

    if coord_start[1] == 0:
        start_ind[1] = 0
    else:
        start_ind[1] = coord_start[1] + overlap

    if coord_start[0] + patch_dim[0] == full_dim[0]:
        end_ind = [full_dim[0], -1]
    else:
        end_ind = [coord_start[0] + patch_dim[0] - overlap, -1]

    if coord_start[1] + patch_dim[1] == full_dim[1]:
        end_ind[1] = full_dim[1]
    else:
        end_ind[1] = coord_start[1] + patch_dim[1] - overlap

    mask[start_ind[0]:end_ind[0], start_ind[1]:end_ind[1]] = 1.0

    return mask


def merge_yx(file_name, out_dim_yx, fov_offset_yx, overlap_mode='avg', tail_dim=False, normalize_modamp=False, write_tiff=False):
    # load patch info
    with open(path.join(FLAGS.save_dir, 'full2patch.json'), 'r') as f:
        info = json.load(f)

    patch = np.load(path.join(FLAGS.save_dir, info[0]['name'], file_name))
    if tail_dim:
        patch = np.swapaxes(np.swapaxes(patch, -1, -3), -1, -2)
    prefix_dim = patch.shape[:-2]
    out_dim_yx = (round(out_dim_yx[0] * FLAGS.zoomfact), round(out_dim_yx[1] * FLAGS.zoomfact))

    print('patch shape: ', patch.shape)
    out = np.zeros(prefix_dim + out_dim_yx, dtype=np.float32)
    if overlap_mode == 'max':
        out -= np.inf
    elif overlap_mode == 'min':
        out += np.inf

    reweigh_map = np.zeros(out_dim_yx, dtype=np.float32)

    for i in range(len(info)):
        patch = np.load(path.join(FLAGS.save_dir, info[i]['name'], file_name))

        if normalize_modamp:
            with np.load(path.join(FLAGS.save_dir, info[i]['name'], 'modamp_init_est.npz')) as modamp_data:
                amp = modamp_data['amp']
                img_avg = modamp_data['img_avg']

            half_margin = np.maximum(int(FLAGS.overlap // 2), 1)
            if np.mean(patch[..., half_margin:-half_margin, half_margin:-half_margin]) > 0 and img_avg > 0:
                patch = patch / np.mean(
                    np.maximum(patch[..., half_margin:-half_margin, half_margin:-half_margin], 0)) * img_avg
            else:
                print('patch {} has non-positive average value'.format(info[i]['name']))
                patch = np.zeros_like(patch)

        if tail_dim:
            patch = np.swapaxes(np.swapaxes(patch, -1, -3), -1, -2)

        coord_start_yx =[c - o for c, o in zip(info[i]['coord_start'][-2:], fov_offset_yx)]
        coord_start_yx[-1] = round(coord_start_yx[-1] * FLAGS.zoomfact)
        coord_start_yx[-2] = round(coord_start_yx[-2] * FLAGS.zoomfact)

        patch_weight = np.zeros_like(reweigh_map)

        if overlap_mode == 'avg':
            patch_weight[coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2], coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]] = 1.0
            out[..., coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2], coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]] += patch
            reweigh_map += patch_weight

        elif overlap_mode == 'max':
            out[..., coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2],
                coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]] = np.maximum(patch, out[..., coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2],
                coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]])

        elif overlap_mode == 'min':
            out[..., coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2],
                coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]] = np.minimum(patch, out[..., coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2],
                coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]])

        elif overlap_mode == 'grad':
            overlap_zoom = round(FLAGS.overlap * FLAGS.zoomfact)
            patch_weight = get_no_overlap_mask_2d(coord_start_yx, patch.shape[-2:], full_dim=out_dim_yx,
                                                  overlap=overlap_zoom)
            patch_weight = convolve2d(patch_weight, np.ones((overlap_zoom+1, overlap_zoom+1)), mode='same',
                                      boundary='symm')**2 / (overlap_zoom+1)**2
            reweigh_map += patch_weight

            patch = patch * patch_weight[coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2],
                            coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]]
            out[...,
                coord_start_yx[0]:coord_start_yx[0] + patch.shape[-2],
                coord_start_yx[1]:coord_start_yx[1] + patch.shape[-1]] += patch

        elif overlap_mode == 'mid':
            assert (FLAGS.overlap % 2 == 0)
            start = (0 if coord_start_yx[0] == 0 else coord_start_yx[0] + FLAGS.overlap//2,
                     0 if coord_start_yx[1] == 0 else coord_start_yx[1] + FLAGS.overlap//2)
            end = (out_dim_yx[0] if coord_start_yx[0] + patch.shape[-2] == out_dim_yx[0] else coord_start_yx[0] + patch.shape[-2] - FLAGS.overlap//2,
                   out_dim_yx[1] if coord_start_yx[1] + patch.shape[-1] == out_dim_yx[1] else coord_start_yx[1] + patch.shape[-1] - FLAGS.overlap//2)
            out[..., start[0]:end[0],start[1]:end[1]] = patch[..., start[0]-coord_start_yx[0]:end[0]-coord_start_yx[0], start[1]-coord_start_yx[1]:end[1]-coord_start_yx[1]]

    if overlap_mode == 'avg' or overlap_mode == 'grad':
        out = out / reweigh_map

    if tail_dim:
        out = np.swapaxes(np.swapaxes(out, -1, -2), -1, -3)

    np.save(path.join(FLAGS.save_dir, file_name), out)

    if write_tiff:
        tif_name = file_name.replace('.npy', '.tif')
        if len(out.shape) == 3 or len(out.shape) == 2:
            tifffile.imwrite(path.join(FLAGS.save_dir, tif_name), out)
        elif len(out.shape) == 4:
            tifffile.imwrite(path.join(FLAGS.save_dir, tif_name), out, imagej=True)


def main(unused_argv):
    full_fov = tuple([int(s) for s in FLAGS.full_fov])

    if FLAGS.fov_offset is not None:
        fov_offset = tuple([int(s) for s in FLAGS.fov_offset])
    else:
        fov_offset = (0, ) * len(full_fov)

    if FLAGS.merge:
        merge_yx('recon.npy', out_dim_yx=full_fov[-2:], fov_offset_yx=fov_offset[-2:], overlap_mode=FLAGS.overlap_mode, tail_dim=False, normalize_modamp=True, write_tiff=FLAGS.write_tiff)
        merge_yx('motion_dense_t.npy', out_dim_yx=full_fov[-2:], fov_offset_yx=fov_offset[-2:], overlap_mode=FLAGS.overlap_mode, tail_dim=True, write_tiff=False)
        merge_yx('recon_dense_t.npy', out_dim_yx=full_fov[-2:], fov_offset_yx=fov_offset[-2:], overlap_mode=FLAGS.overlap_mode, tail_dim=False, normalize_modamp=True, write_tiff=False)
        return

    target_fov = [int(s) for s in FLAGS.target_fov]

    patch_coord_start = cut_patches(full_dim=full_fov, target_dim=target_fov, overlap=FLAGS.overlap).astype(np.int32)
    patch_coord_start = patch_coord_start + np.array(fov_offset)[np.newaxis].astype(np.int32)
    patch_dim = np.tile(np.array(target_fov)[np.newaxis], (patch_coord_start.shape[0], 1)).astype(np.int32)

    patch_ind = np.arange(patch_coord_start.shape[0])
    run_name = FLAGS.save_dir.split('/')[-1]
    patch_name = [run_name + "_{:03d}".format(i) for i in patch_ind]
    patch_path = [path.join(FLAGS.save_dir, name) for name in patch_name]

    info = []
    for i in patch_ind:
        info.append({'ind': int(i), 'name': patch_name[i], 'coord_start': patch_coord_start[i].tolist(),
                     'dim': patch_dim[i].tolist(), 'path': patch_path[i], })

    # check if FLAGS.save_dir exists and make new directory if necessary
    if not path.exists(FLAGS.save_dir):
        os.makedirs(FLAGS.save_dir)

    with open(path.join(FLAGS.save_dir, 'full2patch.json'), 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    define_flags()
    app.run(main)
