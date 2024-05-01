# Description:
#  Jax implementation of the positional encoding for coordinate-based neural networks.
# Written by Ruiming Cao on December 30, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from typing import Union

import jax.numpy as jnp
import jax
from flax.struct import dataclass

import calcil as cc


@dataclass
class PosencParameters:
    posenc_min: int  # the minimum (inclusive) degree of the encoding.
    posenc_max: int  # the maximum (exclusive) degree of the encoding.
    num_freqs: int


class AnnealedPosenc(cc.forward.Model):
    posenc_param: PosencParameters
    dim: Union[jnp.ndarray, None] = None

    def __call__(self, x, alpha=1000):
        """Cat x with a positional encoding of x with scales 2^[min_deg, max_deg-1].

        Instead of computing [sin(x), cos(x)], we use the trig identity
        cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

        From https://github.com/google/mipnerf

        Args:
            x: jnp.ndarray, variables to be encoded. Note that x should be in [-pi, pi].

        Returns:
          encoded: jnp.ndarray, encoded variables.
        """
        if self.dim:
            x = x / self.dim * 2 - 1

        if self.posenc_param.posenc_min == self.posenc_param.posenc_max:
            return x

        freq_bands = 2.0 ** jnp.linspace(self.posenc_param.posenc_min, self.posenc_param.posenc_max, self.posenc_param.num_freqs,
                                         endpoint=False)

        xb = jnp.pi * x[..., None, :] * freq_bands[:, None]

        alpha *= self.posenc_param.num_freqs
        bands = jnp.linspace(self.posenc_param.posenc_min, self.posenc_param.posenc_max, self.posenc_param.num_freqs, endpoint=False)
        coef = jnp.clip(alpha - bands, 0.0, 1.0)
        window = 0.5 * (1 + jnp.cos(jnp.pi * coef + jnp.pi))

        features = jnp.sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1))

        features = window[None, :, None] * features
        features = jnp.reshape(features, list(x.shape[:-1]) + [-1])
        return jnp.concatenate([x, features], axis=-1)
