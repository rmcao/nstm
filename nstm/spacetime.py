# -*- coding: utf-8 -*-
"""Implementation of neural space-time model for 2D/3D+time dynamic scene representations.
"""

import functools
import numpy as np
from flax import linen as nn
import jax.numpy as jnp
import jax
from typing import Callable, Any, Tuple, Union, Dict, List
import calcil as cc

from flax.struct import dataclass
from nstm import utils
from nstm.hash_encoding import HashParameters, HashEmbeddingTime, AnnealedHashEmbedding, HashEmbeddingTimeCombined
from nstm.pos_encoding import AnnealedPosenc, PosencParameters


def generate_dense_yx_coords(dim_yx, normalize=True):
    """Generate a list of coordinates for a 2D grid.

    Args:
        dim_yx (Tuple[int, int]): the dimension of the 2D grid.
        normalize (bool): whether to normalize the coordinates to [-1, 1].

    Returns:
        np.ndarray: the list of coordinates (number of coordinates, [y, x]).
    """
    if normalize:
        xlin = np.arange(dim_yx[1]) / dim_yx[1] * 2 - 1
        ylin = np.arange(dim_yx[0]) / dim_yx[0] * 2 - 1
    else:
        xlin = np.arange(dim_yx[1])
        ylin = np.arange(dim_yx[0])

    y, x = np.meshgrid(ylin, xlin, indexing='ij')
    yx = np.concatenate((y[:, :, None], x[:, :, None]), axis=2)

    return yx.reshape([-1, 2])


def generate_dense_zyx_coords(dim_zyx, start_coord_zyx=None, normalize=True):
    """Generate a list of coordinates for a 3D grid.

    Args:
        dim_zyx (Tuple[int, int, int]): the dimension of the 3D grid.
        start_coord_zyx (Tuple[int, int, int]): the starting coordinate of the 3D grid.
        normalize (bool): whether to normalize the coordinates to [-1, 1].

    Returns:
        np.ndarray: the list of coordinates (number of coordinates, [z, y, x]).
    """
    if normalize and start_coord_zyx:
        raise ValueError('Cannot have custom start coord {} when normalize flag is turned on'.format(start_coord_zyx))

    if start_coord_zyx is None:
        start_coord_zyx = (0, 0, 0)

    if normalize:
        xlin = np.arange(dim_zyx[2]) / dim_zyx[2] * 2 - 1
        ylin = np.arange(dim_zyx[1]) / dim_zyx[1] * 2 - 1
        zlin = np.arange(dim_zyx[0]) / dim_zyx[0] * 2 - 1
    else:
        xlin = np.arange(dim_zyx[2]) + start_coord_zyx[2]
        ylin = np.arange(dim_zyx[1]) + start_coord_zyx[1]
        zlin = np.arange(dim_zyx[0]) + start_coord_zyx[0]

    z, y, x = np.meshgrid(zlin, ylin, xlin, indexing='ij')
    zyx = np.concatenate((z[:, :, :, None], y[:, :, :, None], x[:, :, :, None]), axis=-1)

    return zyx.reshape([-1, 3])


@dataclass
class MLPParameters:
    """Parameters for the MLP model.

    Args:
        net_depth (int): The depth of MLP.
        net_width (int): The width of MLP.
        net_activation (Callable): The activation function.
        skip_layer (int): The layer to add skip layers to.
        kernel_init (Callable): network weight initializer
    """
    net_depth: int
    net_width: int
    net_activation: Callable[..., Any]
    skip_layer: int = 4
    kernel_init: Callable = jax.nn.initializers.glorot_uniform()


@dataclass
class SpaceTimeParameters:
    """Parameters for the space-time model.

    Args:
        motion_mlp_param (MLPParameters): Parameters for the motion network.
        object_mlp_param (MLPParameters): Parameters for the scene network.
        motion_embedding (Union[str, None]): The type of input embedding for motion network.
        motion_embedding_param (Union[Dict, HashParameters, PosencParameters]):
            Parameters for the motion network's input embedding.
        object_embedding (Union[str, None]): The type of input embedding for scene network.
        object_embedding_param (Union[Dict, HashParameters, PosencParameters]):
            Parameters for the object network's input embedding.
        out_activation (Callable[..., Any]): The activation function for MLP output.
    """

    motion_mlp_param: MLPParameters
    object_mlp_param: MLPParameters
    motion_embedding: Union[str, None]
    motion_embedding_param: Union[Dict, HashParameters, PosencParameters]
    object_embedding: Union[str, None]
    object_embedding_param: Union[Dict, HashParameters, PosencParameters]
    out_activation: Callable[..., Any]


class MLP(cc.forward.Model):
    """A simple MLP with an option to add skip connections.

    Args:
        net_depth (int): The depth of the MLP.
        net_width (int): The width of the MLP.
        net_activation (Callable): The activation function.
        skip_layer (int): The layer to add skip layers to.
        num_output_channels (int): The number of output channels.
        kernel_init (Callable): weight initializer
        precision (nn.linear.PrecisionLike): arithmetic precision of the network
        param_dtype (jnp.dtype): data type of the parameters
    """
    net_depth: int = 8
    net_width: int = 256
    net_activation: Callable = nn.relu
    skip_layer: int = 4
    num_output_channels: int = 1
    kernel_init: Callable = jax.nn.initializers.glorot_uniform()
    precision: nn.linear.PrecisionLike = None
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        """Evaluate the MLP.

        Args:
          x (jnp.ndarray): list of input features to evaluate, [batch, num_samples, feature].

        Returns:
          jnp.ndarray: MLP output values, [batch, num_samples, num_outpuut_channels].
        """
        input_dim = x.shape[:-1]
        feature_dim = x.shape[-1]
        x = x.reshape([-1, feature_dim])
        dense_layer = functools.partial(
            nn.Dense, kernel_init=self.kernel_init, precision=self.precision, param_dtype=self.param_dtype)
        inputs = x

        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)

        output = dense_layer(self.num_output_channels)(x).reshape(
            input_dim + (self.num_output_channels, ))
        return output


class SpaceTimeMLP(cc.forward.Model):
    """Implementation of neural space-time model.

    The model takes the spatial and temporal coordinate as an input, and outputs the object's
    properties at the given time and spatial location.

    Args:
        optical_param (Union[utils.SystemParameters, utils.SystemParameters3D, Tuple[int, int], Tuple[int, int, int]):
            the optical parameters to specify matrix size. Can alternatively directly specify the matrix size using a
            tuple of (y,x) or (z,y,x).
        spacetime_param (SpaceTimeParameters): the space-time parameters.
        num_output_channels (int): the number of output channels.
        precision (str, optional): the arithmetic precision of the model.
    """
    optical_param: Union[utils.SystemParameters, utils.SystemParameters3D, Tuple[int, int], Tuple[int, int, int]]
    spacetime_param: SpaceTimeParameters
    num_output_channels: int
    precision: str = 'float32'

    def setup(self):
        if self.precision == 'float32':
            precision = 'float32'
            self.param_dtype = jnp.float32
        elif self.precision == 'float16':
            precision = 'bfloat16'
            self.param_dtype = jnp.bfloat16
        else:
            raise NotImplementedError(f'precision = {self.precision} not implemented.')

        if isinstance(self.optical_param, tuple):
            if len(self.optical_param) == 2:
                self.ndim = 2
            elif len(self.optical_param) == 3:
                self.ndim = 3
            else:
                raise ValueError('Wrong input for optical_param.')

            self.dim_x = self.optical_param
        else:
            if 'dim_yx' in self.optical_param.__annotations__:
                self.ndim = 2
            elif 'dim_zyx' in self.optical_param.__annotations__:
                self.ndim = 3
            else:
                raise ValueError('Wrong input for optical_param.')

            if self.ndim == 3:
                self.dim_x = (self.optical_param.dim_zyx[0] + self.optical_param.padding_zyx[0]*2,
                              self.optical_param.dim_zyx[1] + self.optical_param.padding_zyx[1]*2,
                              self.optical_param.dim_zyx[2] + self.optical_param.padding_zyx[2]*2)
            else:
                self.dim_x = (self.optical_param.dim_yx[0] + self.optical_param.padding_yx[0]*2,
                              self.optical_param.dim_yx[1] + self.optical_param.padding_yx[1]*2)

        # motion MLP
        self.motion_mlp = MLP(net_depth=self.spacetime_param.motion_mlp_param.net_depth,
                              net_width=self.spacetime_param.motion_mlp_param.net_width,
                              net_activation=self.spacetime_param.motion_mlp_param.net_activation,
                              skip_layer=self.spacetime_param.motion_mlp_param.skip_layer,
                              num_output_channels=self.ndim,
                              kernel_init=self.spacetime_param.motion_mlp_param.kernel_init,
                              precision=precision,
                              param_dtype=self.param_dtype)

        # object MLP
        self.object_mlp = MLP(net_depth=self.spacetime_param.object_mlp_param.net_depth,
                              net_width=self.spacetime_param.object_mlp_param.net_width,
                              net_activation=self.spacetime_param.object_mlp_param.net_activation,
                              skip_layer=self.spacetime_param.object_mlp_param.skip_layer,
                              num_output_channels=self.num_output_channels,
                              kernel_init=self.spacetime_param.object_mlp_param.kernel_init,
                              precision=precision,
                              param_dtype=self.param_dtype)

        # motion embedding
        if self.spacetime_param.motion_embedding == 'hash':
            self.motion_embedding = HashEmbeddingTime(self.spacetime_param.motion_embedding_param['space'],
                                                      self.spacetime_param.motion_embedding_param['time'])
        elif self.spacetime_param.motion_embedding == 'hash_combined':
            self.motion_embedding = HashEmbeddingTimeCombined(self.spacetime_param.motion_embedding_param,
                                                              precision=self.precision)
        elif self.spacetime_param.motion_embedding == 'posenc':
            self.time_posenc = AnnealedPosenc(self.spacetime_param.motion_embedding_param)
            self.motion_embedding = lambda xt, alpha: jnp.concatenate([
                xt[..., :self.ndim],self.time_posenc(xt[..., -1:], alpha=alpha)], axis=-1)
        elif self.spacetime_param.motion_embedding is None:
            self.motion_embedding = lambda xt, alpha: xt
        else:
            raise NotImplementedError(f'Time embedding: {self.spacetime_param.motion_embedding} is not implemented.')

        # object embedding
        if self.spacetime_param.object_embedding == 'hash':
            self.object_embedding = AnnealedHashEmbedding(hash_param=self.spacetime_param.object_embedding_param,
                                                          n_input_features=self.ndim, precision=self.precision)
        elif self.spacetime_param.object_embedding == 'posenc':
            self.object_embedding = AnnealedPosenc(self.spacetime_param.object_embedding_param,
                                                   jnp.array([[self.dim_x]]))
            raise NotImplementedError

        elif self.spacetime_param.motion_embedding is None:
            self.object_embedding = lambda x, alpha: x / jnp.array([[self.dim_x]]) * 2 - 1
        else:
            raise NotImplementedError(f'Object embedding: {self.spacetime_param.object_embedding} is not implemented.')

        if self.ndim == 3:
            self.list_coord = jnp.asarray(generate_dense_zyx_coords(self.dim_x, start_coord_zyx=(0, 0, 0),
                                                                    normalize=False)[np.newaxis, :, :],
                                          dtype=self.param_dtype)
        elif self.ndim == 2:
            self.list_coord = jnp.asarray(generate_dense_yx_coords(self.dim_x, normalize=False)[jnp.newaxis, :, :],
                                          dtype=self.param_dtype)

    def __call__(self, t: jnp.ndarray, coord_offset: jnp.ndarray, alpha: float = 1e5):
        """
        Forward pass for the space-time model. The model takes a list of timestamps and outputs a list of objects at the
        given timestamps.

        Args:
            t (jnp.ndarray): the list of timestamps to query the space-time model, [batch].
            coord_offset (jnp.ndarray): the offset to the spatial coordinates, [batch, (2 or 3)].
            alpha (float): the annealing parameter for the positional encoding.

        Returns:
            jnp.ndarray: the list of objects at the given timestamps, 2D - [batch, y, x, num_output_channels],
                        3D - [batch, z, y, x, num_output_channels].
        """
        list_zyx = jnp.tile(self.list_coord, (t.shape[0], 1, 1))
        list_zyx = list_zyx + jnp.asarray(coord_offset[:, np.newaxis, :], dtype=self.param_dtype)
        list_zyxt = jnp.concatenate([list_zyx,
                                    jnp.tile(jnp.asarray(t[:, np.newaxis, np.newaxis], dtype=self.param_dtype), (1, list_zyx.shape[1], 1))], axis=-1)

        list_zyxt_embedded = self.motion_embedding(list_zyxt, alpha=alpha)
        list_zyx_time_adjusted = self.motion_mlp(list_zyxt_embedded) * jnp.array(self.dim_x)[jnp.newaxis, jnp.newaxis, :] + list_zyx

        list_zyx_embedded = self.object_embedding(list_zyx_time_adjusted, alpha=alpha)

        output = self.object_mlp(list_zyx_embedded)

        # reshape MLP's output
        output = jnp.reshape(output, (-1, ) + self.dim_x + (self.num_output_channels, ))
        output = self.spacetime_param.out_activation(output)

        return output

    def get_motion_map(self, t: jnp.ndarray, coord_offset: jnp.ndarray, alpha: float = 1e5):
        """Get the pixel/voxel-level motion map at the given timestamps.

        Args:
            t (jnp.ndarray): the list of timestamps to query the space-time model, [batch].
            coord_offset (jnp.ndarray): the offset to the spatial coordinates, [batch, (2 or 3)].
            alpha (float): the annealing parameter for the positional encoding.

        Returns:
            jnp.ndarray: the motion map at the given timestamps, 2D - [batch, y, x, 2], 3D - [batch, z, y, x, 3].
        """
        list_zyx = jnp.tile(self.list_coord, (t.shape[0], 1, 1))
        list_zyx = list_zyx + coord_offset[:, np.newaxis, :]

        list_zyxt = jnp.concatenate([list_zyx,
                                    jnp.tile(t[:, np.newaxis, np.newaxis], (1, list_zyx.shape[1], 1))], axis=-1)

        list_zyxt_embedded = self.motion_embedding(list_zyxt, alpha=alpha)
        motion_zyx = self.motion_mlp(list_zyxt_embedded) * jnp.array(self.dim_x)[jnp.newaxis, jnp.newaxis, :]

        return motion_zyx.reshape((-1, ) + self.dim_x + (self.ndim, ))


def get_trajectory(motion_zyx: np.ndarray,
                   target_pts: Union[List[Tuple[int, int]], List[Tuple[int, int, int]]],
                   interpolate: bool = False):
    """Helper function to get the trajectory of the target points based on the motion map generated by motion network.

    Args:
        motion_zyx (np.ndarray): a list of motion maps at different timepoints.
            2D - [num_frames, y, x, 2], 3D - [num_frames, z, y, x, 3].
        target_pts (Union[List[Tuple[int, int]], List[Tuple[int, int, int]]]):
            the list of target points to track. 2D - [(y, x),]*num_targets, 3D - [(z, y, x)]*num_targets.
        interpolate (bool): whether to interpolate the trajectory.

    Returns:
        List[np.ndarray]: the list of trajectories for the target points.
            2D - [num_targets, num_frames, 2], 3D - [num_targets, num_frames, 3].
    """
    ndim = motion_zyx.shape[-1]
    dim_x = motion_zyx.shape[1:-1]
    num_frames = motion_zyx.shape[0]

    pos_t_zyx = generate_dense_yx_coords(dim_x, normalize=False).reshape((1,) + dim_x + (ndim, ))
    pos_t_zyx = motion_zyx + pos_t_zyx

    list_trajectories = []

    for target_x_f0 in target_pts:
        if ndim == 2:
            target_x = np.array([pos_t_zyx[0, target_x_f0[0], target_x_f0[1], 0],
                                 pos_t_zyx[0, target_x_f0[0], target_x_f0[1], 1]])
        elif ndim == 3:
            target_x = np.array([pos_t_zyx[0, target_x_f0[0], target_x_f0[1], target_x_f0[2], 0],
                                 pos_t_zyx[0, target_x_f0[0], target_x_f0[1], target_x_f0[2], 1],
                                 pos_t_zyx[0, target_x_f0[0], target_x_f0[1], target_x_f0[2], 2],])
        else:
            raise NotImplementedError(f'ndim: {ndim} is not implemented.')

        trajectory = []
        for i in range(num_frames):
            if interpolate:
                dist = np.sum(np.where((pos_t_zyx[i] - target_x) > 0, pos_t_zyx[i] - target_x, 1e7), axis=-1)
                ind = np.unravel_index(np.argmin(dist), dim_x)

                # check boundary:
                if dist[ind] >= 5:
                    trajectory.append(np.unravel_index(np.argmin(np.sum((pos_t_zyx[i] - target_x) ** 2, axis=-1)), dim_x))
                    continue

                if ndim == 2:
                    next_pos_zyx = pos_t_zyx[i][ind[0], ind[1]]
                    cur_pos_zyx = pos_t_zyx[i][ind[0]-1, ind[1]-1]
                elif ndim == 3:
                    next_pos_zyx = pos_t_zyx[i][ind[0], ind[1], ind[2]]
                    cur_pos_zyx = pos_t_zyx[i][ind[0]-1, ind[1]-1, ind[2]-1]
                else:
                    raise NotImplementedError(f'ndim: {ndim} is not implemented.')

                k = np.clip((target_x - next_pos_zyx) / (cur_pos_zyx - next_pos_zyx), 0, 1)

                trajectory.append(np.array(ind)-1 + (1 - k))
            else:
                trajectory.append(np.unravel_index(np.argmin(np.sum((pos_t_zyx[i] - target_x) ** 2, axis=-1)), dim_x))
        list_trajectories.append(np.array(trajectory))

    return list_trajectories
