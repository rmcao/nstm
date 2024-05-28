# sim3d_main.py - Description:
#  Main script for 3D SIM reconstruction with neural-space time model.
# Created by Ruiming Cao on Apr 23, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io
import os
import glob
import warnings

from absl import app
from absl import flags
from jax.config import config
import json
import jax
import numpy as np
from flax import linen as nn
import tifffile

import calcil as cc

from nstm import utils
from nstm import sim3d_utils
from nstm import datasets
from nstm import sim3d_flow
from nstm import spacetime

FLAGS = flags.FLAGS


def main(unused_argv):
    if FLAGS.config is not None:
        utils.update_flags(FLAGS)

    rng = jax.random.PRNGKey(FLAGS.seed)
    padding_zyx = (FLAGS.padding_z, FLAGS.padding, FLAGS.padding)

    if FLAGS.raw_path.lower().endswith('.npz'):
        save_path = FLAGS.save_path
        with np.load(os.path.join(FLAGS.data_dir, FLAGS.raw_path)) as d:
            timestamp_phase = d['timestamp_phase']
            img = d['img']
            OTF = d['OTF']

        dim_zyx = img.shape[2:]
        dim_otf_zx = [OTF.shape[2], (OTF.shape[1] - 1) * 2]
    else:
        if FLAGS.patch_json:
            # load patch info
            with open(os.path.join(FLAGS.patch_json), 'r') as f:
                patches = json.load(f)
                patch = patches[FLAGS.patch_ind]

            coord_start = [int(s) for s in patch['coord_start']]
            save_path = patch['path'] if FLAGS.save_path is None else FLAGS.save_path
            patch_dim_zyx = [int(s) for s in patch['dim']]
        else:
            coord_start = [int(s) for s in FLAGS.coord_start]
            save_path = FLAGS.save_path
            patch_dim_zyx = [int(s) for s in FLAGS.patch_dim]

        # set up parameters
        fov_zyxshw = (coord_start[0], coord_start[1], coord_start[2],
                      patch_dim_zyx[0], patch_dim_zyx[1], patch_dim_zyx[2])
        assert ((1.0 * fov_zyxshw[4] * FLAGS.zoomfact).is_integer() and
                (1.0 * fov_zyxshw[5] * FLAGS.zoomfact).is_integer()), \
            "The zoom factor * dim must be an integer.{}, {}".format(fov_zyxshw[4] * FLAGS.zoomfact,
                                                                     fov_zyxshw[5] * FLAGS.zoomfact)
        dim_zyx = (fov_zyxshw[3], int(fov_zyxshw[4] * FLAGS.zoomfact), int(fov_zyxshw[5] * FLAGS.zoomfact))
        dim_otf_zx = [int(s) for s in FLAGS.dim_otf_zx]

    # save FLAGS into a file
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if not FLAGS.eval_only:
        with open(os.path.join(save_path, 'FLAGS.json'), 'w') as f:
            json.dump(FLAGS.flag_values_dict(), f, indent=4)

    optical_param = utils.SystemParameters3D(dim_zyx, FLAGS.wavelength, FLAGS.wavelength_exc, FLAGS.na,
                                             FLAGS.ps / FLAGS.zoomfact, FLAGS.dz, FLAGS.ri_medium, padding_zyx)
    otf_param = utils.SystemParameters3D((dim_otf_zx[0], dim_otf_zx[1], dim_otf_zx[1]),
                                         FLAGS.wavelength, FLAGS.wavelength_exc, FLAGS.na, FLAGS.ps_otf,
                                         FLAGS.dz_otf, FLAGS.ri_medium, (0, 0, 0))
    line_spacing = [float(s) for s in FLAGS.line_spacing]
    k0angles = [float(s) for s in FLAGS.k0angles]

    if not FLAGS.raw_path.lower().endswith('.npz'):
        # load & processed images
        otf_path = sorted(glob.glob(os.path.join(FLAGS.data_dir, FLAGS.otf_path)))
        d = datasets.SIM3DDataLoaderMultitime(os.path.join(FLAGS.data_dir, FLAGS.otf_path),
                                              os.path.join(FLAGS.data_dir, FLAGS.meta_path),
                                              FLAGS.zoomfact, FLAGS.ndirs, FLAGS.nphases,
                                              num_timepoint=FLAGS.num_stack,
                                              start_timepoint=FLAGS.starting_stack)
        img = d.load_3DSIM_raw(os.path.join(FLAGS.data_dir, FLAGS.raw_path), fov_zyxshw,
                               background_int=FLAGS.background_int, normalize=False)

        OTF = d.load_3DSIM_OTF(otf_path)

        # load metadata
        timestamp_phase, _ = d.load_metadata(normalize=True, avg_phase=False, single_plane_time=FLAGS.fast_mode)

    img_avg = np.mean(img)
    img_std = np.std(img)
    img_max = np.max(img)
    img_min = np.min(img)
    img = img / np.max(img)

    if FLAGS.lr_init_motion == 0:
        warnings.warn('lr_init_motion is set to 0, '
                      'which means no time dependency is applied and all input timepoints are zero.')
        timestamp_phase = np.zeros_like(timestamp_phase)

    bandplus_img = np.array([sim3d_utils.separate_bands(img_, out_positive_bands=True) for img_ in img])

    # OTF loading
    sim_param, phase, amp = sim3d_utils.get_otf(bandplus_img[:, :, :optical_param.dim_zyx[0]], OTF, optical_param,
                                                otf_param, FLAGS.ndirs, FLAGS.nphases, k0angles, line_spacing,
                                                crop_boundary_zyx=(2 if optical_param.dim_zyx[0] > 10 else 0, 20, 20),
                                                noisy=True, notch=False)
    otf_valid_region, otf_valid_region_rfft = sim3d_utils.otf_support_mask(optical_param, otf_param, sim_param, OTF[0])

    np.savez(os.path.join(save_path, 'modamp_init_est.npz'), phase=phase, amp=amp,
             img_avg=img_avg, img_std=img_std, img_max=img_max, img_min=img_min,)
    print('saving modamp parameters to ' + os.path.join(save_path, 'modamp_init_est.npz'))

    # data loader
    if FLAGS.fast_mode:
        assert FLAGS.num_stack == 1, 'fast_mode only currently supports single acquisition timepoint.'

        ind_k0angle = np.tile(np.arange(FLAGS.ndirs)[:, np.newaxis], (1, FLAGS.nphases))
        ind_phases = np.tile(np.arange(FLAGS.nphases)[np.newaxis, :], (FLAGS.ndirs, 1))
        z_offset = np.ones((FLAGS.ndirs, FLAGS.nphases)) * (optical_param.dim_zyx[0] // 2 + optical_param.padding_zyx[0])
        zyx_offset = np.stack([z_offset, np.zeros_like(z_offset), np.zeros_like(z_offset)], axis=-1)

        data_loader = cc.data_utils.loader_from_numpy(
            {'img': img.reshape((-1,) + optical_param.dim_zyx),
             't': timestamp_phase.reshape(-1),
             'ind_phase': ind_phases.reshape((-1)),
             'ind_k0angle': ind_k0angle.reshape(-1),
             'zyx_offset': zyx_offset.reshape((-1, 3))},
            prefix_dim=(FLAGS.batch_size,), seed=85471)
    else:
        assert optical_param.padding_zyx[0] == 0, ('padding_zyx[0] must be 0 for non-fast mode. '
                                                   'It is wasteful to use padding.')
        z_mask = np.zeros((optical_param.dim_zyx[0]))
        z_mask[optical_param.dim_zyx[0] // 2] = 1
        z_mask = np.tile(z_mask[np.newaxis], (FLAGS.batch_size, 1))
        ind_k0angle = np.tile(np.arange(FLAGS.ndirs)[:, np.newaxis, np.newaxis],
                              (1, FLAGS.nphases, optical_param.dim_zyx[0] * FLAGS.num_stack))
        ind_phases = np.tile(np.arange(FLAGS.nphases)[np.newaxis, :, np.newaxis],
                             (FLAGS.ndirs, 1, optical_param.dim_zyx[0] * FLAGS.num_stack))
        z_offset = np.tile(np.arange(optical_param.dim_zyx[0])[np.newaxis, np.newaxis, :],
                           (FLAGS.ndirs, FLAGS.nphases, FLAGS.num_stack))
        zyx_offset = np.stack([z_offset, np.zeros_like(z_offset), np.zeros_like(z_offset)], axis=-1)

        data_loader = cc.data_utils.loader_from_numpy({'img': img.reshape((-1, optical_param.dim_zyx[1], optical_param.dim_zyx[2])),
                                                       't': timestamp_phase.reshape(-1),
                                                       'ind_phase': ind_phases.reshape((-1)),
                                                       'ind_k0angle': ind_k0angle.reshape(-1),
                                                       'zyx_offset': zyx_offset.reshape((-1, 3))},
                                                      prefix_dim=(FLAGS.batch_size,), seed=85471,
                                                      aux_terms={'z_mask': z_mask})

    sample_epoch_input = next(data_loader)
    num_batches_per_epoch = len(sample_epoch_input)
    sample_input_dict = sample_epoch_input[0]

    # model initialization
    model = define_model(optical_param=optical_param, sim_param=sim_param)
    if not FLAGS.eval_only:
        variables = model.init(rng, input_dict=sample_input_dict)

    # loss functions
    nonneg_reg = cc.loss.Loss(sim3d_flow.gen_loss_nonneg_reg(), 'nonneg_reg', has_intermediates=True)
    if FLAGS.fast_mode:
        l2_loss = cc.loss.Loss(sim3d_flow.gen_loss_l2_stack(margin=FLAGS.l2_loss_margin), 'l2')
    else:
        l2_loss = cc.loss.Loss(sim3d_flow.gen_loss_l2(margin=FLAGS.l2_loss_margin), 'l2')
    total_loss = l2_loss + nonneg_reg * float(FLAGS.nonneg_reg_w)

    # render function for tensorboard image logging
    def render_fn(variables, train_state):
        out = [model.apply(variables, np.array([0.]), np.array([[int(np.median(z_offset.reshape(-1))), 0, 0]]),
                           method=lambda module, t, b: module.spacetime(t, b)[
                               0, optical_param.dim_zyx[0] // 2, ..., 0])]
        return {'recon_fluo': out}

    # set up optimization
    recon_param = cc.reconstruction.ReconIterParameters(save_dir=save_path, n_epoch=FLAGS.num_epoch,
                                                        keep_checkpoints=10,
                                                        checkpoint_every=FLAGS.save_every,
                                                        output_every=FLAGS.render_every, log_every=FLAGS.print_every,)
    var_params = define_training_params(num_batches_per_epoch=num_batches_per_epoch)

    if FLAGS.eval_only:
        # load reconstructed variables
        recon_variables = cc.reconstruction.load_checkpoint_and_output(save_path)
    else:
        # run reconstruction
        recon_variables, recon = cc.reconstruction.reconstruct_multivars_sgd(model.apply, variables, var_params,
                                                                             data_loader, total_loss, recon_param,
                                                                             render_fn)

    # save reconstructed frames and motion map
    recon_dense_t = np.zeros((optical_param.dim_zyx[0], FLAGS.ndirs, FLAGS.nphases, optical_param.dim_zyx[1],
                              optical_param.dim_zyx[2])).astype(np.float32)
    motion_dense_t = np.zeros((optical_param.dim_zyx[0], FLAGS.ndirs, FLAGS.nphases, optical_param.dim_zyx[1],
                               optical_param.dim_zyx[2], 3)).astype(np.float32)

    for i_phase in range(FLAGS.nphases):
        for i_dir in range(FLAGS.ndirs):
            if FLAGS.fast_mode:
                recon_dense_t[:, i_dir, i_phase] = model.apply(
                    recon_variables, np.array([timestamp_phase[i_dir, i_phase]]),
                    method=lambda module, t: module.spacetime(
                        t, np.array([[z_offset[0, 0], 0, 0]]))[0, optical_param.padding_zyx[0]:optical_param.padding_zyx[0]+optical_param.dim_zyx[0], :, :, 0])

                motion_dense_t[:, i_dir, i_phase] = model.apply(
                    recon_variables, np.array([timestamp_phase[i_dir, i_phase]]),
                    method=lambda module, a: module.spacetime.get_motion_map(
                        a, np.array([[z_offset[0, 0], 0, 0]]))[0])[..., optical_param.padding_zyx[0]:optical_param.padding_zyx[0]+optical_param.dim_zyx[0], :, :, :]
            else:
                for i_z in range(optical_param.dim_zyx[0]):
                    recon_dense_t[i_z, i_dir, i_phase] = model.apply(
                        recon_variables, np.array([timestamp_phase[i_dir, i_phase, i_z]]),
                        method=lambda module, t: module.spacetime(
                            t, np.array([[optical_param.dim_zyx[0] // 2, 0, 0]]))[0, i_z, :, :, 0])

                    motion_dense_t[i_z, i_dir, i_phase] = model.apply(
                        recon_variables, np.array([timestamp_phase[i_dir, i_phase, i_z]]),
                        method=lambda module, a: module.spacetime.get_motion_map(
                            a, np.array([[optical_param.dim_zyx[0] // 2, 0, 0]]))[0, i_z])
    np.save(os.path.join(save_path, 'recon_dense_t.npy'), recon_dense_t)
    np.save(os.path.join(save_path, 'motion_dense_t.npy'), motion_dense_t)

    if optical_param.dim_zyx[0] > 1:
        recon_out = recon_dense_t[:, :, 2]
    else:
        recon_out = recon_dense_t.reshape((1, FLAGS.ndirs * FLAGS.nphases, optical_param.dim_zyx[1], optical_param.dim_zyx[2]))

    recon_out = (
        np.fft.ifftn(otf_valid_region[:, np.newaxis, :, :] * np.fft.fftn(recon_out, axes=(-4, -2, -1)), axes=(-4, -2, -1)).real).astype(
        np.float32)
    tifffile.imwrite(os.path.join(save_path, 'recon_filtered.tif'), recon_out, imagej=True)
    np.save(os.path.join(save_path, 'recon.npy'), recon_out)

    recon_out_fft = np.log10(np.abs(np.fft.fftshift(np.fft.fftn(recon_dense_t[:, 0, 2], axes=(-3, -2, -1)))))
    tifffile.imwrite(os.path.join(save_path, 'recon_fft.tif'), recon_out_fft)

    bf_img = np.mean(img.astype(np.float32), axis=1).transpose((1, 0, 2, 3))
    tifffile.imwrite(os.path.join(save_path, 'bf.tif'), bf_img, imagej=True)

    print("Final mod amp matrix: ")
    print(recon_variables['params']['fluo_forward'])


def define_model(optical_param, sim_param):
    assert FLAGS.object_hash_boundary_x >= 0, "object_hash_boundary_x must be non-negative."

    # set up object net hash parameter based on ratio of object size
    object_hash_fine = np.array([FLAGS.object_hash_ratio_z * optical_param.dim_zyx[0],
                                 FLAGS.object_hash_ratio * optical_param.dim_zyx[1] / FLAGS.zoomfact,
                                 FLAGS.object_hash_ratio * optical_param.dim_zyx[2] / FLAGS.zoomfact])
    object_hash_base = object_hash_fine * np.array([0.3, 0.1, 0.1])
    if optical_param.dim_zyx[0] == 1:
        object_hash_base[0] = 1
        object_hash_fine[0] = 1

    # set up motion net hash parameter based on ratio of fine object hash
    motion_hash_fine = np.array([object_hash_fine[0] * FLAGS.motion_hash_ratio_z,
                                 object_hash_fine[1] * FLAGS.motion_hash_ratio,
                                 object_hash_fine[2] * FLAGS.motion_hash_ratio,
                                 FLAGS.motion_hash_temporal])
    motion_hash_base = motion_hash_fine * np.array([1./3, 0.1, 0.1, 1])
    motion_hash_base[3] = 1
    if optical_param.dim_zyx[0] == 1:
        motion_hash_base[0] = 1
        motion_hash_fine[0] = 1

    if FLAGS.object_hash_base is not None:
        print('object_hash_base is specified to: ', FLAGS.object_hash_base)
        object_hash_base = np.array([float(s) for s in FLAGS.object_hash_base])
    if FLAGS.object_hash_fine is not None:
        print('object_hash_fine is specified to: ', FLAGS.object_hash_fine)
        object_hash_fine = np.array([float(s) for s in FLAGS.object_hash_fine])

    object_hash_base[1:] *= (1 + FLAGS.object_hash_boundary_x)
    object_hash_fine[1:] *= (1 + FLAGS.object_hash_boundary_x)

    if FLAGS.motion_hash_base is not None:
        print('motion_hash_base is specified to: ', FLAGS.motion_hash_base)
        motion_hash_base = [float(s) for s in FLAGS.motion_hash_base]
    if FLAGS.motion_hash_fine is not None:
        print('motion_hash_fine is specified to: ', FLAGS.motion_hash_fine)
        motion_hash_fine = [float(s) for s in FLAGS.motion_hash_fine]

    print('Hash embedding for object net: base - ', object_hash_base, ' fine - ', object_hash_fine)
    print('Hash embedding for motion net: base - ', motion_hash_base, ' fine - ', motion_hash_fine)

    # set up model parameters
    hash_param = spacetime.HashParameters(
        bounding_box=(np.array([0, -optical_param.dim_zyx[1]*FLAGS.object_hash_boundary_x/2, -optical_param.dim_zyx[2]*FLAGS.object_hash_boundary_x/2]),
                      np.array([optical_param.dim_zyx[0] * 2, optical_param.dim_zyx[1] * (1 + FLAGS.object_hash_boundary_x/2), optical_param.dim_zyx[2] * (1 + FLAGS.object_hash_boundary_x/2)])),
        n_levels=8, n_features_per_level=2, log2_hashmap_size=16, base_resolution=object_hash_base,
        finest_resolution=object_hash_fine)
    hash_param_motion_space = spacetime.HashParameters(
        bounding_box=(np.array([0, 0, 0]), np.array([optical_param.dim_zyx[0] * 2, optical_param.dim_zyx[1], optical_param.dim_zyx[2]])),
        n_levels=8, n_features_per_level=2, log2_hashmap_size=16, base_resolution=np.array(motion_hash_base[:3]),
        finest_resolution=np.array(motion_hash_fine[:3]))
    hash_param_motion_time = spacetime.HashParameters(bounding_box=(np.array([-1]), np.array([1])), n_levels=8,
                                            n_features_per_level=6,  # 2
                                            log2_hashmap_size=10, base_resolution=motion_hash_base[3],
                                            finest_resolution=motion_hash_fine[3] * FLAGS.num_stack)
    hash_param_motion_spacetime = spacetime.HashParameters(
        bounding_box=(np.array([0, 0, 0, -1]), np.array([optical_param.dim_zyx[0] * 2, optical_param.dim_zyx[1], optical_param.dim_zyx[2], 1])),
        n_levels=8, n_features_per_level=2, log2_hashmap_size=16, base_resolution=np.array(motion_hash_base),
        finest_resolution=np.array(list(motion_hash_fine[:3]) + [motion_hash_fine[3] * FLAGS.num_stack,]))

    motion_embedding_param = {'space': hash_param_motion_space, 'time': hash_param_motion_time}

    object_mlp_param = spacetime.MLPParameters(net_depth=FLAGS.object_net_depth, net_width=FLAGS.object_net_width,
                                               net_activation=nn.gelu, skip_layer=4)
    motion_mlp_param = spacetime.MLPParameters(net_depth=FLAGS.motion_net_depth, net_width=FLAGS.motion_net_width,
                                               net_activation=nn.elu, skip_layer=4)

    if FLAGS.motion_hash == 'combined':
        space_time_param = spacetime.SpaceTimeParameters(motion_mlp_param=motion_mlp_param,
                                                         object_mlp_param=object_mlp_param,
                                                         motion_embedding='hash_combined',
                                                         motion_embedding_param=hash_param_motion_spacetime,
                                                         object_embedding='hash', object_embedding_param=hash_param,
                                                         out_activation=(lambda x: x) if (FLAGS.object_out_activation is None) else getattr(nn, FLAGS.object_out_activation))
    else:
        space_time_param = spacetime.SpaceTimeParameters(motion_mlp_param=motion_mlp_param,
                                                         object_mlp_param=object_mlp_param,
                                                         motion_embedding='hash',
                                                         motion_embedding_param=motion_embedding_param,
                                                         object_embedding='hash', object_embedding_param=hash_param,
                                                         out_activation=(lambda x: x) if (FLAGS.object_out_activation is None) else getattr(nn, FLAGS.object_out_activation))

    model = sim3d_flow.SIM3DSpacetime(
        sim_param, space_time_param, optical_param,
        annealed_epoch=FLAGS.num_epoch * FLAGS.annealed_rate,
        order0_grad_reduction=FLAGS.band0_grad_reduction)

    return model


def define_training_params(num_batches_per_epoch):
    trans_steps = num_batches_per_epoch * FLAGS.num_epoch
    no_update_params = cc.reconstruction.ReconVarParameters(lr=0)

    if FLAGS.modamp_delay is None:
        modamp_delay = FLAGS.annealed_rate
    else:
        modamp_delay = FLAGS.modamp_delay

    object_mlp_params = cc.reconstruction.ReconVarParameters(lr=float(FLAGS.lr_init_object), opt='adam',
                                                             opt_kwargs={'b1': 0.9, 'b2': 0.99, 'eps': 1e-15},
                                                             schedule='exponential',
                                                             schedule_kwargs={'transition_steps': trans_steps,
                                                                              'decay_rate': FLAGS.lr_decay_object,
                                                                              'transition_begin': 0,
                                                                              'end_value': FLAGS.lr_final},
                                                             update_every=FLAGS.update_every_object)
    motion_mlp_params = cc.reconstruction.ReconVarParameters(lr=float(FLAGS.lr_init_motion), opt='adam',
                                                             opt_kwargs={'b1': 0.9, 'b2': 0.99, 'eps': 1e-15},
                                                             schedule='exponential',
                                                             schedule_kwargs={'transition_steps': trans_steps,
                                                                              'decay_rate': FLAGS.lr_decay_motion,
                                                                              'transition_begin': 0,
                                                                              'end_value': FLAGS.lr_final},
                                                             update_every=FLAGS.update_every_motion,
                                                             delay_update_n_iter=0)
    update_params_modamp = cc.reconstruction.ReconVarParameters(lr=float(FLAGS.lr_init_modamp), opt='adam',
                                                                schedule='exponential',
                                                                schedule_kwargs={'transition_steps': 1e4,
                                                                                 'decay_rate': 0.1,
                                                                                 'transition_begin': 0},
                                                                update_every=num_batches_per_epoch,
                                                                delay_update_n_iter=trans_steps * modamp_delay)

    var_params = {'params': {'spacetime': {'motion_mlp': motion_mlp_params, 'object_mlp': object_mlp_params,
                                           'motion_embedding': motion_mlp_params,
                                           'object_embedding': object_mlp_params},
                             'fluo_forward': update_params_modamp}}
    return var_params


if __name__ == "__main__":
    sim3d_utils.define_flags()
    config.parse_flags_with_absl()
    app.run(main)