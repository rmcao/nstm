# dpc_flow.py - Description:
#  Differential phase contrast with space-time modeling
# Created by Ruiming Cao on May 08, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

import numpy as np
import jax.numpy as jnp

import calcil as cc
from nstm import utils
from nstm import dpc_utils
from nstm import spacetime


class DPC(cc.forward.Model):
    optical_param: utils.SystemParameters
    list_source: np.ndarray
    precision: str = 'float32'
    """
    DPC - Description:
        Differential phase contrast forward model
        
    Args:
        optical_param (utils.SystemParameters): Optical parameters of the system.
        list_source (np.ndarray): List of illumination patterns used for the DPC system.
        precision (str, optional): Precision of the model. Defaults to 'float32'.
    """


    def setup(self):
        pupil = cc.physics.wave_optics.genPupilNumpy(self.optical_param.dim_yx, self.optical_param.pixel_size,
                                                     self.optical_param.na, self.optical_param.wavelength)
        self.Hu, self.Hp = dpc_utils.gen_transfer_func(list_source=self.list_source, pupil=pupil,
                                                       wavelength=self.optical_param.wavelength, shifted_out=False)

    def __call__(self, absorption, phase):
        out = jnp.fft.ifft2((self.Hu[jnp.newaxis] * jnp.fft.fft2(absorption, axes=(-2, -1))[:, jnp.newaxis]) +
                            (self.Hp[jnp.newaxis] * jnp.fft.fft2(phase, axes=(-2, -1))[:, jnp.newaxis])).real

        return out


class DPCFlow(cc.forward.Model):
    optical_param: utils.SystemParameters
    list_source: np.ndarray
    spacetime_param: spacetime.SpaceTimeParameters
    annealed_epoch: float = 1
    phase_only: bool = False
    precision: str = 'float32'
    """
    DPCFlow - Description:
        Differential phase contrast with space-time modeling
    
    Args:
        optical_param (utils.SystemParameters): Optical parameters of the system.
        list_source (np.ndarray): List of illumination patterns used for the DPC system.
        spacetime_param (spacetime.SpaceTimeParameters): Space-time modeling parameters.
        annealed_epoch (float, optional): The number of annealed epochs for coarse-to-fine optimization. 
                                          Defaults to 1, i.e., no coarse-to-fine.
        phase_only (bool, optional): Whether to use phase-only input. Defaults to False.
        precision (str, optional): Precision of the model. Defaults to 'float32'.
    """

    def setup(self):
        self.spacetime = spacetime.SpaceTimeMLP(self.optical_param,
                                                self.spacetime_param,
                                                num_output_channels=2,
                                                precision=self.precision)
        self.forward = DPC(self.optical_param, self.list_source)

    def __call__(self, input_dict):
        """
        Forward pass of the DPCFlow model.

        Args:
            input_dict:

        Returns:
            out: The rendered measurements corresponding to the input timepoints.
        """
        absorp_phase = self.spacetime(input_dict['t'], jnp.zeros((1, 2)),
                                      alpha=input_dict['epoch']/self.annealed_epoch)
        self.sow('intermediates', 'absorp_phase', absorp_phase)

        if self.phase_only:
            out = self.forward(jnp.zeros_like(absorp_phase[..., 0]), absorp_phase[..., 1])
        else:
            out = self.forward(absorp_phase[..., 0], absorp_phase[..., 1])

        return out


def gen_loss_l2(margin=0):
    """Returns the L2 loss function for the DPC model."""
    assert margin >= 0, "the spatial margin needs to be non-negative."
    if margin == 0:
        def loss_l2(forward_output, variables, input_dict, intermediates):
            l2 = (jnp.abs(input_dict['img'] - jnp.sum(forward_output * input_dict['ind_pat'][:, :, np.newaxis, np.newaxis], axis=1)) ** 2).mean()
            return l2

    else:
        def loss_l2(forward_output, variables, input_dict, intermediates):
            l2 = (jnp.abs(input_dict['img'] - jnp.sum(forward_output * input_dict['ind_pat'][:, :, np.newaxis, np.newaxis], axis=1))[:, margin:-margin, margin:-margin] ** 2).mean()
            return l2

    return loss_l2


def gen_l2_reg_absorp(freq_space=False):
    """Returns the L2 regularization function for absorption term."""
    def l2_reg_absorp(forward_output, variables, input_dict, intermediates):
        if freq_space:
            l2_reg = jnp.mean(jnp.abs(jnp.fft.fft2(intermediates['absorp_phase'][-1][..., 0]))**2)
        else:
            l2_reg = jnp.mean(jnp.maximum(-intermediates['absorp_phase'][-1][..., 0], 0.0)**2)

        return l2_reg

    return l2_reg_absorp


def gen_l2_reg_phase(freq_space=False):
    """Returns the L2 regularization function for phase term."""
    def l2_reg_phase(forward_output, variables, input_dict, intermediates):
        if freq_space:
            l2_reg = jnp.mean(jnp.abs(jnp.fft.fft2(intermediates['absorp_phase'][-1][..., 1]))**2)
        else:
            l2_reg = jnp.mean(jnp.abs(intermediates['absorp_phase'][-1][..., 1])**2)
        return l2_reg
    return l2_reg_phase
