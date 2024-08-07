# -*- coding: utf-8 -*-
"""Main script for rolling-shutter DiffuserCam reconstruction with neural space-time model."""

import os
import numpy as np
from absl import app
from absl import flags
from jax.config import config
import jax
import jax.numpy as jnp
from flax import linen as nn
import calcil as cc
import imageio

from nstm import utils
from nstm import diffcam_utils
from nstm import diffcam_flow
from nstm import spacetime
from nstm.hash_encoding import HashParameters

FLAGS = flags.FLAGS


def main(unused_argv):
    if FLAGS.config is not None:
        utils.update_flags(FLAGS)

    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)

    img, psf = diffcam_utils.load_data_psf(FLAGS.raw_path, FLAGS.psf_path, FLAGS.background_int, FLAGS.ds)
    dim_yx = (psf.shape[0], psf.shape[1])

    nlines = int(np.round(FLAGS.exposure_time / FLAGS.readout_time / FLAGS.ds))

    if FLAGS.roll_pixels_x != 0:
        img = np.roll(img, FLAGS.roll_pixels_x, axis=1)

    # set up shutter indicator
    pad_fn = lambda x: np.pad(x, ((psf.shape[0] // 2, psf.shape[0] // 2), (psf.shape[1] // 2, psf.shape[1] // 2)))
    shutter_ind = diffcam_utils.gen_indicator((dim_yx[0], dim_yx[1]), nlines, pad_fn,
                                              downsample_t=False)

    object_hash_base = [float(s) for s in FLAGS.object_hash_base]
    object_hash_fine = [float(s) for s in FLAGS.object_hash_fine]
    motion_hash_base = [float(s) for s in FLAGS.motion_hash_base]
    motion_hash_fine = [float(s) for s in FLAGS.motion_hash_fine]
    hash_param = HashParameters(bounding_box=(np.array([0, 0]), np.array([dim_yx[0] // 2, dim_yx[1] // 2])),
                                n_levels=8, n_features_per_level=2, log2_hashmap_size=16,
                                base_resolution=np.array(object_hash_base),
                                finest_resolution=np.array(object_hash_fine))
    hash_param_motion_spacetime = HashParameters(
        bounding_box=(np.array([0, 0, -1]), np.array([dim_yx[0] // 2, dim_yx[1] // 2, 1])),
        n_levels=8, n_features_per_level=2, log2_hashmap_size=16,
        base_resolution=np.array(motion_hash_base),
        finest_resolution=np.array(motion_hash_fine))
    object_mlp_param = spacetime.MLPParameters(net_depth=FLAGS.object_net_depth, net_width=FLAGS.object_net_width,
                                               net_activation=getattr(nn, FLAGS.object_act_fn), skip_layer=4)
    motion_mlp_param = spacetime.MLPParameters(net_depth=FLAGS.motion_net_depth, net_width=FLAGS.motion_net_width,
                                               net_activation=getattr(nn, FLAGS.motion_act_fn), skip_layer=4)

    spacetime_param = spacetime.SpaceTimeParameters(motion_mlp_param=motion_mlp_param,
                                                    object_mlp_param=object_mlp_param,
                                                    motion_embedding='hash_combined',
                                                    motion_embedding_param=hash_param_motion_spacetime,
                                                    object_embedding='hash', object_embedding_param=hash_param,
                                                    out_activation=nn.elu)

    tmp = shutter_ind.transpose((2, 0, 1))[:, dim_yx[0] // 2:-dim_yx[0] // 2, dim_yx[1] // 2:-dim_yx[1] // 2]
    shutter_ind_sum = np.sum(tmp[:, :, tmp.shape[2] // 2], axis=0)
    row_mask = np.zeros((tmp.shape[0] - FLAGS.num_lines_per_forward + 1, tmp.shape[1]), dtype=bool)
    time_mask = np.zeros((tmp.shape[0] - FLAGS.num_lines_per_forward + 1, FLAGS.num_lines_per_forward), dtype=np.int32)
    for i in range(tmp.shape[0] - FLAGS.num_lines_per_forward + 1):
        row_mask[i] = np.sum(tmp[i:i + FLAGS.num_lines_per_forward, :, tmp.shape[2] // 2], axis=0) == shutter_ind_sum
        time_mask[i] = np.arange(FLAGS.num_lines_per_forward, dtype=np.int32) + i

    data_loader = cc.data_utils.loader_from_numpy({'mask': row_mask, 't_mask': time_mask},
                                                  prefix_dim=(FLAGS.batch_size,),
                                                  aux_terms={'img': img},
                                                  seed=85471,)
    list_sample_input_dict = next(data_loader)
    num_steps_per_epoch = len(list_sample_input_dict)

    model = diffcam_flow.DiffuserCamRSFlow(psf, nlines, spacetime_param,
                                           annealed_epoch=FLAGS.annealed_rate * FLAGS.num_epoch,
                                           ram_efficient=True)
    rng = jax.random.PRNGKey(0)
    variables = model.init(rng, input_dict=list_sample_input_dict[0])

    l2 = cc.loss.Loss(diffcam_flow.gen_loss_l2_row(), 'l2')
    nonneg = cc.loss.Loss(diffcam_flow.gen_nonneg_reg(), 'nonneg', has_intermediates=True)
    total_loss = l2 + float(FLAGS.nonneg_reg_w) * nonneg

    def output_fn(v, train_state):
        out = model.apply(v, jnp.array([0]),
                          method=lambda module, a: module.spacetime(a, np.zeros((1, 2))))
        return {'img': [out[0, ..., 0]]}

    num_steps = num_steps_per_epoch * FLAGS.num_epoch

    recon_param = cc.reconstruction.ReconIterParameters(save_dir=FLAGS.save_path, n_epoch=FLAGS.num_epoch,
                                                        keep_checkpoints=5,
                                                        checkpoint_every=FLAGS.save_every,
                                                        output_every=FLAGS.render_every, log_every=FLAGS.print_every, )

    mlp_params = cc.reconstruction.ReconVarParameters(lr=float(FLAGS.lr_init_object), opt='adam',
                                                      opt_kwargs={'b1': 0.9, 'b2': 0.99, 'eps': 1e-15},
                                                      schedule='exponential',
                                                      schedule_kwargs={'transition_steps': num_steps, 'decay_rate': 0.1,
                                                                       'transition_begin': 0},
                                                      update_every=FLAGS.update_every_object)
    motion_mlp_params = cc.reconstruction.ReconVarParameters(lr=float(FLAGS.lr_init_motion), opt='adam',
                                                             opt_kwargs={'b1': 0.9, 'b2': 0.99, 'eps': 1e-15},
                                                             schedule='exponential',
                                                             schedule_kwargs={'transition_steps': num_steps,
                                                                              'decay_rate': 0.1,
                                                                              'transition_begin': 0},
                                                             update_every=FLAGS.update_every_motion)

    var_params = {'params': {'spacetime': {'motion_mlp': motion_mlp_params, 'object_mlp': mlp_params,
                                           'motion_embedding': motion_mlp_params, 'object_embedding': mlp_params}, }}

    recon_variables, recon = cc.reconstruction.reconstruct_multivars_sgd(model.apply, variables, var_params,
                                                                         data_loader, total_loss, recon_param,
                                                                         output_fn, None)

    time = np.reshape(np.arange(shutter_ind.shape[2]) / shutter_ind.shape[2] * 2 - 1, (-1, 10))
    inference = jax.jit(lambda t: model.apply(recon_variables, jnp.array(t),
                                              method=lambda module, a: module.spacetime(a, jnp.array([[0, 0]]))))
    recon_t = np.array([inference(t) for t in time])
    recon_t = recon_t / np.max(recon_t)
    recon_t = recon_t.reshape((-1, dim_yx[0] // 2, dim_yx[1] // 2, 3))
    np.save(os.path.join(FLAGS.save_path, 'recon_t.npy'), recon_t)

    imageio.mimwrite(os.path.join(FLAGS.save_path, 'recon.gif'), (np.maximum(recon_t, 0.0) * 255).astype(np.uint8))

    inference_motion = jax.jit(lambda t: model.apply(
        recon_variables, jnp.array(t),
        method=lambda module, a: module.spacetime.get_motion_map(a, jnp.array([[0, 0]]))))
    motion_t = np.array([inference_motion(t) for t in time])
    motion_t = motion_t.reshape((-1, dim_yx[0]//2, dim_yx[1]//2, 2))
    np.save(os.path.join(FLAGS.save_path, 'motion_t.npy'), motion_t)


if __name__ == "__main__":
    diffcam_utils.define_flags()
    config.parse_flags_with_absl()
    app.run(main)
