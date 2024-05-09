# diffcam_flow.py - Description:
#  Rolling shutter diffuserCam reconstruction with neural space-time model.
# Created by Ruiming Cao on May 25, 2023
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from typing import Tuple
import numpy as np
import jax.numpy as jnp
import jax
import calcil as cc
import utils
import diffcam_utils
import spacetime


class DiffuserCanRS(cc.forward.Model):
    dim_yx: Tuple[int, int]
    psf: jnp.ndarray
    nlines: int
    downsample_t: bool

    def setup(self):
        pad_fn = lambda x: np.pad(x, ((self.dim_yx[0] // 2, self.dim_yx[0] - self.dim_yx[0] // 2),
                                      (self.dim_yx[1] // 2, self.dim_yx[1] - self.dim_yx[1] // 2)))
        self.shutter_indicator = jnp.array(diffcam_utils.gen_indicator((self.dim_yx[0], self.dim_yx[1]),
                                                             self.nlines, pad_fn, downsample_t=self.downsample_t).transpose((2, 0, 1))[:,
                               self.dim_yx[0] // 2:-self.dim_yx[0] // 2, self.dim_yx[1] // 2:-self.dim_yx[1] // 2])
        self.dim_t = self.shutter_indicator.shape[0]
        self.psf_pad = jnp.pad(self.psf, ((self.dim_yx[0] // 2, self.dim_yx[0] - self.dim_yx[0] // 2),
                                          (self.dim_yx[1] // 2, self.dim_yx[1] - self.dim_yx[1] // 2), (0, 0)))
        self.f_psf_pad = jnp.fft.rfft2(jnp.fft.ifftshift(self.psf_pad, axes=(-3, -2)), axes=(-3, -2))

    def __call__(self, x):
        x = jnp.fft.irfft2(jnp.fft.rfft2(x, axes=(-3, -2)) * self.f_psf_pad, axes=(-3, -2))[
            :, self.dim_yx[0] // 2:-self.dim_yx[0] // 2, self.dim_yx[1] // 2:-self.dim_yx[1] // 2, ...]

        x = x * self.shutter_indicator[..., jnp.newaxis]
        y = jnp.sum(x, axis=0)

        return y

    def efficient(self, x, t_mask):
        x = jnp.fft.irfft2(jnp.fft.rfft2(x, axes=(-3, -2)) * self.f_psf_pad, axes=(-3, -2))[
            :, self.dim_yx[0] // 2:-self.dim_yx[0] // 2, self.dim_yx[1] // 2:-self.dim_yx[1] // 2, ...]

        x = x * self.shutter_indicator[t_mask][..., jnp.newaxis]
        y = jnp.sum(x, axis=0)

        return y


class DiffuserCamRSFlow(cc.forward.Model):
    psf: jnp.ndarray
    nlines: int
    spacetime_param: spacetime.SpaceTimeParameters
    annealed_epoch: float = 1
    ram_efficient: bool = False

    def setup(self):
        self.dim_yx = (self.psf.shape[0], self.psf.shape[1])

        optical_param = utils.SystemParameters((self.dim_yx[0]//2, self.dim_yx[1]//2), wavelength=0, na=0, pixel_size=0, RI_medium=0)
        self.spacetime = spacetime.SpaceTimeMLP(optical_param,
                                                self.spacetime_param,
                                                num_output_channels=3)

        self.forward = DiffuserCanRS(self.dim_yx, self.psf, self.nlines, downsample_t=not self.ram_efficient)
        self.t_all = jnp.arange(self.forward.dim_t) / self.forward.dim_t * 2 - 1

    def __call__(self, input_dict):
        if self.ram_efficient:
            x = self.spacetime(t=jnp.asarray(input_dict['t_mask'][0]/self.forward.dim_t * 2-1), coord_offset=jnp.zeros((1, 2)),
                               alpha=input_dict['epoch'] / self.annealed_epoch)
            self.sow('intermediates', 'obj', x)

            x_pad = jnp.pad(x, ((0, 0), (
            self.dim_yx[0] // 2 + self.dim_yx[0] // 4, self.dim_yx[0] // 2 + self.dim_yx[0] // 2 - self.dim_yx[0] // 4),
                                (self.dim_yx[1] // 2 + self.dim_yx[1] // 4,
                                 self.dim_yx[1] // 2 + self.dim_yx[1] // 2 - self.dim_yx[1] // 4), (0, 0)))
            y = self.forward.efficient(x_pad, input_dict['t_mask'][0])
        else:
            x = self.spacetime(t=self.t_all, coord_offset=jnp.zeros((1, 2)), alpha=input_dict['epoch']/self.annealed_epoch)
            x_pad = jnp.pad(x, ((0, 0), (self.dim_yx[0]//2 + self.dim_yx[0]//4, self.dim_yx[0]//2 + self.dim_yx[0]//2 - self.dim_yx[0]//4),
                                (self.dim_yx[1]//2 + self.dim_yx[1]//4, self.dim_yx[1]//2 + self.dim_yx[1]// 2 - self.dim_yx[1]//4), (0, 0)))
            y = self.forward(x_pad)
        return y


def gen_loss_l2():
    def loss_l2(forward_output, variables, input_dict, intermediates):
        l2 = ((input_dict['img'] - forward_output) ** 2).mean()
        return l2
    return loss_l2


def gen_loss_l2_row():
    def loss_l2(forward_output, variables, input_dict, intermediates):
        l2 = (((input_dict['img'] - forward_output) * input_dict['mask'][0, :, jnp.newaxis, jnp.newaxis]) ** 2).mean()
        return l2
    return loss_l2


def gen_nonneg_reg():
    def loss_nonneg_reg(forward_output, variables, input_dict, intermediates):
        nonneg = -jnp.minimum(intermediates['obj'][-1], 0.0).mean()
        return nonneg
    return loss_nonneg_reg
