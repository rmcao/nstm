# -*- coding: utf-8 -*-
"""3D SIM forward model with neural-space time model.

This module contains the forward models for 3D SIM and 3D SIM with space-time modeling. The loss functions used for the
3D SIM reconstruction are also provided.
"""

from typing import Tuple, Union
import numpy as np
import jax.numpy as jnp
import jax
from flax import linen as nn

import calcil as cc
from nstm import utils
from nstm import sim3d_utils
from nstm import spacetime
from nstm.sim3d_utils import SIMParameter3D


class FluoSIM3D(cc.forward.Model):
    """This function mimicks the setting and forward model used in the original 3D SIM paper (Eq.9) in Biophysical
    Journal 94(12) 4957–4970."""
    sim_param: SIMParameter3D
    optical_param: utils.SystemParameters3D
    order0_grad_reduction: float = 0.0  # Reduce the gradient related to band 0 in the forward model (range 0-1)
    apo_filter: bool = True

    def setup(self):
        self.phases = np.linspace(0, np.pi*2, self.sim_param.nphases, endpoint=False)

        self.illum = jnp.array([[[sim3d_utils.generate_sinusoidal(
            self.optical_param,
            k0angle=self.sim_param.k0angles[i_k0angle] + np.pi,
            k0mag=1 / self.sim_param.line_spacing[i_k0angle],
            phase=self.sim_param.starting_phases[i_k0angle][o] - phase,
            order=o) for o in range(3)]
            for phase in self.phases] for i_k0angle in range(self.sim_param.ndirs)])

        self.OTF = jnp.asarray(self.sim_param.OTF)  # [dir, band, fz, fy, fx]
        self.modamp = self.param('modamp', nn.initializers.ones, (self.sim_param.ndirs, 3, 1, 1, 1), )
        self.apo = utils.apodization(self.optical_param, k0mag=1/min(self.sim_param.line_spacing), norder=3,
                                     inverted=True, min_height=1e-2)

    def __call__(self, fluo_density, ind_k0angle, ind_phase):
        if self.apo_filter:
            fluo_density = jnp.fft.ifftn(jnp.fft.fftn(fluo_density, axes=(-3, -2, -1)) * self.apo, axes=(-3, -2, -1))

        out_bands = self.illum[ind_k0angle, ind_phase, :, jnp.newaxis] * fluo_density[..., jnp.newaxis, :, :, :]

        out_bands = jnp.fft.fftn(out_bands, axes=(-3, -2, -1)) * self.OTF[ind_k0angle]

        out_bands = jnp.fft.ifftn(out_bands, axes=(-3, -2, -1))[...,
                    self.optical_param.padding_zyx[0]:self.optical_param.padding_zyx[0]+self.optical_param.dim_zyx[0],
                    self.optical_param.padding_zyx[1]:self.optical_param.padding_zyx[1]+self.optical_param.dim_zyx[1],
                    self.optical_param.padding_zyx[2]:self.optical_param.padding_zyx[2]+self.optical_param.dim_zyx[2]]

        if self.order0_grad_reduction > 0.0:
            out_bands = out_bands.at[..., 0, :, :, :].set(
                jax.lax.stop_gradient(out_bands[..., 0, :, :, :]) * self.order0_grad_reduction +
                out_bands[..., 0, :, :, :] * (1 - self.order0_grad_reduction))

        out = jnp.sum(out_bands.real * jnp.abs(self.modamp[ind_k0angle]) / jnp.sum(jnp.abs(self.modamp[ind_k0angle]))
                      * 3.0, axis=-4)

        return out


class FluoSIM3DWrapper(cc.forward.Model):
    sim_param: SIMParameter3D
    optical_param: utils.SystemParameters3D

    def setup(self):
        self.fluo_density = self.param('fluo_matrix', nn.initializers.zeros, self.optical_param.dim_zyx)
        self.fluo_3DSIM = FluoSIM3D(self.sim_param, self.optical_param, order0_grad_reduction=0.8)
        self.k0angle_indices = np.tile(np.arange(self.sim_param.ndirs)[:, np.newaxis], (1, self.sim_param.nphases))
        self.phase_indices = np.tile(np.arange(self.sim_param.nphases)[np.newaxis, :], (self.sim_param.ndirs, 1))

    def __call__(self, input_dict):
        out = self.fluo_3DSIM(self.fluo_density, self.k0angle_indices.reshape(-1), self.phase_indices.reshape(-1))
        return out.reshape((self.sim_param.ndirs, self.sim_param.nphases,) + self.optical_param.dim_zyx)


class SIM3DSpacetime(cc.forward.Model):
    sim_param: SIMParameter3D
    spacetime_param: spacetime.SpaceTimeParameters
    optical_param: utils.SystemParameters3D
    annealed_epoch: float = 1
    order0_grad_reduction: float = 0.0

    def setup(self):
        self.spacetime = spacetime.SpaceTimeMLP(self.optical_param,
                                                self.spacetime_param,
                                                num_output_channels=1)

        self.fluo_forward = FluoSIM3D(self.sim_param, self.optical_param,
                                      order0_grad_reduction=self.order0_grad_reduction)

    def __call__(self, input_dict):
        fluo_density = self.spacetime(t=input_dict['t'],
                                      coord_offset=input_dict['zyx_offset'],
                                      alpha=input_dict['epoch']/self.annealed_epoch)[..., 0]

        self.sow('intermediates', 'fluo', fluo_density)

        out = self.fluo_forward(fluo_density, input_dict['ind_k0angle'], input_dict['ind_phase'])

        return out


def gen_loss_nonneg_reg():
    def loss_nonneg_reg(forward_output, variables, input_dict, intermediates):
        loss_nonneg = -jnp.minimum(intermediates['fluo'][-1], 0.0).mean()
        return loss_nonneg
    return loss_nonneg_reg


def gen_loss_l2(margin=1):
    if margin > 0:
        def loss_l2_fn(forward_output, variables, input_dict, intermediates):
            loss_l2 = (jnp.abs(input_dict['img'] - jnp.sum(forward_output * input_dict['z_mask'][:, :, jnp.newaxis, jnp.newaxis], axis=1))**2)[:, margin:-margin, margin:-margin].mean()
            return loss_l2
    else:
        def loss_l2_fn(forward_output, variables, input_dict, intermediates):
            loss_l2 = (jnp.abs(
                input_dict['img'] - jnp.sum(forward_output * input_dict['z_mask'][:, :, jnp.newaxis, jnp.newaxis],
                                            axis=1)) ** 2).mean()
            return loss_l2
    return loss_l2_fn


def gen_loss_l2_stack(margin=1):
    if margin > 0:
        def loss_l2_stack(forward_output, variables, input_dict, intermediates):
            loss_l2 = (jnp.abs(input_dict['img'] - forward_output)**2)[:, :, margin:-margin, margin:-margin].mean()
            return loss_l2
    else:
        def loss_l2_stack(forward_output, variables, input_dict, intermediates):
            loss_l2 = (jnp.abs(input_dict['img'] - forward_output) ** 2).mean()
            return loss_l2
    return loss_l2_stack
