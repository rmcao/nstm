# Description:
#  Jax implementation of hash encoding for accelerated implicit neural representation for 1D-4D scenes.
#  Our implementation is based on the paper:
#    Müller, Thomas, et al. "Instant neural graphics primitives with a multiresolution hash encoding." ACM transactions
#     on graphics (TOG) 41.4 (2022): 1-15.
#  with a lot of references from this pytorch implementation:
#    https://github.com/yashbhalgat/HashNeRF-pytorch/blob/main/utils.py
#
# Written by Ruiming Cao on November 23, 2022
# Contact: rcao@berkeley.edu
# Website: https://rmcao.github.io

from typing import Tuple, Union
from functools import partial

import numpy as np
import jax.numpy as jnp
import jax
from flax import linen as nn
from flax.struct import dataclass

import calcil as cc


BOX_OFFSETS_4D = jnp.asarray([[[i,j,k, l] for i in [0, 1] for j in [0, 1] for k in [0, 1] for l in [0, 1]]])
BOX_OFFSETS_3D = jnp.asarray([[[i,j,k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]])
BOX_OFFSETS_2D = jnp.asarray([[[i,j] for i in [0, 1] for j in [0, 1]]])
BOX_OFFSETS_1D = jnp.asarray([[[i] for i in [0, 1]]])


def precision_to_dtype(precision):
    if precision == 'float32':
        return jnp.float32
    elif precision == 'float16':
        return jnp.bfloat16
    else:
        raise NotImplementedError(f'precision = {precision} not implemented.')


@dataclass
class HashParameters:
    bounding_box: Tuple[jnp.ndarray, jnp.ndarray]  # (jnp.array([starting coord for each dim]), jnp.array([ending coord for each dim]) )
    n_levels: int = 16
    n_features_per_level: int = 2
    log2_hashmap_size: int = 19
    base_resolution: Union[int, np.ndarray] = 16
    finest_resolution: Union[int, np.ndarray] = 512
    init_uniform_std: float = 1e-4


class HashEmbedding(cc.forward.Model):
    hash_param: HashParameters
    n_input_features: int = -1  # supports [1, 2, 3, 4]. -1 for inferring from bounding box
    precision: str = 'float32'

    def setup(self) -> None:
        self.param_dtype = precision_to_dtype(self.precision)
        self.out_dim = self.hash_param.n_levels * self.hash_param.n_features_per_level

        self.b = jnp.asarray(np.exp((np.log(self.hash_param.finest_resolution) - np.log(self.hash_param.base_resolution)) / (self.hash_param.n_levels - 1))).astype(self.param_dtype)

        self.embeddings = self.param('hash',
                                     nn.initializers.uniform(self.hash_param.init_uniform_std, dtype=self.param_dtype),
                                     (self.hash_param.n_levels, 2 ** self.hash_param.log2_hashmap_size, self.hash_param.n_features_per_level))

        if self.n_input_features == -1:
            self.n_input_features_ = len(self.hash_param.bounding_box[0])
        else:
            self.n_input_features_ = self.n_input_features

        assert (self.n_input_features_ == len(self.hash_param.bounding_box[0])), f"n_input_features={self.n_input_features} does not match bounding_box={self.hash_param.bounding_box}."

        self.box_dim = jnp.asarray([1.0 for _ in range(self.n_input_features_)], dtype=self.param_dtype)

        if self.n_input_features_ == 1:
            self.box_offsets = BOX_OFFSETS_1D.astype(self.param_dtype)
            self.interp_fn = linear_interp
        elif self.n_input_features_ == 2:
            self.box_offsets = BOX_OFFSETS_2D.astype(self.param_dtype)
            self.interp_fn = bilinear_interp
        elif self.n_input_features_ == 3:
            self.box_offsets = BOX_OFFSETS_3D.astype(self.param_dtype)
            self.interp_fn = trilinear_interp
        elif self.n_input_features_ == 4:
            self.box_offsets = BOX_OFFSETS_4D.astype(self.param_dtype)
            self.interp_fn = quadrilinear_interp
        else:
            raise NotImplementedError(f'n_input_features = {self.n_input_features} not implemented.')

    def __call__(self, x):
        input_dim = x.shape[:-1]
        feature_dim = x.shape[-1]
        x = x.reshape([-1, feature_dim])

        # x is 1D-4D point position: B x [1-4]
        x_embedded_all = jax.vmap(self.encoding_at_level, in_axes=[None, 0], out_axes=-1)(x, jnp.arange(self.hash_param.n_levels))

        return jnp.reshape(x_embedded_all, (input_dim + (self.hash_param.n_levels * self.hash_param.n_features_per_level, )))

    def encoding_at_level(self, x, level):
        resolution = jnp.maximum(jnp.floor(self.hash_param.base_resolution * self.b ** level), 1.0)
        pixel_min_vertex, pixel_max_vertex, hashed_pixel_indices = get_pixel_vertices(
            x, jnp.asarray(self.hash_param.bounding_box, dtype=self.param_dtype), resolution, self.hash_param.log2_hashmap_size, self.box_offsets, self.box_dim)

        pixel_embedds = self.embeddings[level][hashed_pixel_indices]

        x_embedded = self.interp_fn(x, pixel_min_vertex, pixel_max_vertex, pixel_embedds)
        return x_embedded


class AnnealedHashEmbedding(cc.forward.Model):
    hash_param: HashParameters
    n_input_features: int = -1  # supports [1, 2, 3]. -1 for inferring from bounding box
    precision: str = 'float32'

    def setup(self) -> None:
        dtype = precision_to_dtype(self.precision)
        self.hash_embedding = HashEmbedding(hash_param=self.hash_param,
                                            n_input_features=self.n_input_features,
                                            precision=self.precision)
        self.bands = jnp.asarray(np.tile(np.arange(self.hash_param.n_levels)[:, np.newaxis],
                             (1, self.hash_param.n_features_per_level)).reshape((-1))).astype(dtype)

    def __call__(self, x, alpha):
        x_embedded = self.hash_embedding(x)
        coef = jnp.clip(alpha * self.hash_param.n_levels - self.bands, 0.0, 1.0)
        window = 0.5 * (1 + jnp.cos(jnp.pi * coef + jnp.pi))

        x_embedded = x_embedded * window[None, :]
        return x_embedded


class HashEmbeddingTime(cc.forward.Model):
    hash_param_space: Union[HashParameters, None]
    hash_param_time: HashParameters
    precision: str = 'float32'

    def setup(self) -> None:
        if self.hash_param_space:
            self.space_embedding = AnnealedHashEmbedding(hash_param=self.hash_param_space,
                                                         n_input_features=-1,
                                                         precision=self.precision)
        else:
            self.space_embedding = lambda x: x

        self.time_embedding = AnnealedHashEmbedding(hash_param=self.hash_param_time,
                                                    n_input_features=1,
                                                    precision=self.precision)

    def __call__(self, yxt, alpha):
        x_embedded = self.space_embedding(yxt[..., :-1], alpha)
        t_embedded = self.time_embedding(yxt[..., -1:], alpha)
        return jnp.concatenate([x_embedded, t_embedded], axis=-1)


class HashEmbeddingTimeCombined(cc.forward.Model):
    hash_param_spacetime: HashParameters
    precision: str = 'float32'

    def setup(self) -> None:
        self.embedding = AnnealedHashEmbedding(hash_param=self.hash_param_spacetime,
                                               n_input_features=-1,
                                               precision=self.precision)

    def __call__(self, xt, alpha):
        xt_embedded = self.embedding(xt, alpha)
        return xt_embedded


def hash_fn(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''

    # hack to enforce 32-bit int. not sure about the performance
    # primes = np.array([1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737], dtype=np.int32)
    primes = jnp.array([1, 1431655781, 622729787, 338294347, 1183186591], dtype=jnp.uint32)
    # xor_result = jax.lax.fori_loop(0, coords.shape[-1], lambda i, val: (val^(coords[..., i]*primes[i])).astype(jnp.uint32),
    #                                jnp.zeros_like(coords.astype(jnp.uint32))[..., 0])
    xor_result = jnp.zeros_like(coords)[..., 0].astype(jnp.uint32)
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i]*primes[i]

    return (1<<log2_hashmap_size)-1 & xor_result


def get_pixel_vertices(x, bounding_box, resolution, log2_hashmap_size, box_offsets, box_dim):
    '''
    x: 1-3D coordinates of samples. B x [1-3]
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max = bounding_box

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = jnp.floor((x - box_min) / grid_size).astype(jnp.uint32)
    pixel_min_vertex = bottom_left_idx * grid_size + box_min
    pixel_max_vertex = pixel_min_vertex + box_dim * grid_size

    pixel_indices = (jnp.expand_dims(bottom_left_idx, 1) + box_offsets).astype(jnp.uint32)

    hashed_pixel_indices = hash_fn(pixel_indices, log2_hashmap_size)

    return pixel_min_vertex, pixel_max_vertex, hashed_pixel_indices


def linear_interp(x, pixel_min_vertex, pixel_max_vertex, pixel_embedds):
    weights = (x - pixel_min_vertex) / (pixel_max_vertex - pixel_min_vertex)  # B x 1

    c = pixel_embedds[:, 0, :] * (1 - weights[:, 0, jnp.newaxis]) + pixel_embedds[:, 1, :] * weights[:, 0, jnp.newaxis]

    return c


def bilinear_interp(x, pixel_min_vertex, pixel_max_vertex, pixel_embedds):

    weights = (x - pixel_min_vertex)/(pixel_max_vertex-pixel_min_vertex) # B x 2

    # step 1
    c0 = pixel_embedds[:,0,:]*(1-weights[:,0,jnp.newaxis]) + pixel_embedds[:,2,:]*weights[:,0,jnp.newaxis]
    c1 = pixel_embedds[:,1,:]*(1-weights[:,0,jnp.newaxis]) + pixel_embedds[:,3,:]*weights[:,0,jnp.newaxis]

    # step 2
    c = c0*(1-weights[:,1,jnp.newaxis]) + c1*weights[:,1,jnp.newaxis]

    return c


def trilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: batch x 3
    voxel_min_vertex: batch x 3
    voxel_max_vertex: batch x 3
    voxel_embedds: batch x 8 x num_feature
    '''
    # source: https://en.wikipedia.org/wiki/Trilinear_interpolation
    weights = (x - voxel_min_vertex)/(voxel_max_vertex-voxel_min_vertex) # B x 3

    # step 1
    # 0->000, 1->001, 2->010, 3->011, 4->100, 5->101, 6->110, 7->111
    c00 = voxel_embedds[:,0]*(1-weights[:,0][:,jnp.newaxis]) + voxel_embedds[:,4]*weights[:,0][:,jnp.newaxis]
    c01 = voxel_embedds[:,1]*(1-weights[:,0][:,jnp.newaxis]) + voxel_embedds[:,5]*weights[:,0][:,jnp.newaxis]
    c10 = voxel_embedds[:,2]*(1-weights[:,0][:,jnp.newaxis]) + voxel_embedds[:,6]*weights[:,0][:,jnp.newaxis]
    c11 = voxel_embedds[:,3]*(1-weights[:,0][:,jnp.newaxis]) + voxel_embedds[:,7]*weights[:,0][:,jnp.newaxis]

    # step 2
    c0 = c00*(1-weights[:,1][:,jnp.newaxis]) + c10*weights[:,1][:,jnp.newaxis]
    c1 = c01*(1-weights[:,1][:,jnp.newaxis]) + c11*weights[:,1][:,jnp.newaxis]

    # step 3
    c = c0*(1-weights[:,2][:,jnp.newaxis]) + c1*weights[:,2][:,jnp.newaxis]

    return c


def quadrilinear_interp(x, voxel_min_vertex, voxel_max_vertex, voxel_embedds):
    '''
    x: batch x 4
    voxel_min_vertex: batch x 4
    voxel_max_vertex: batch x 4
    voxel_embedds: batch x 16 x num_feature
    '''
    # Generated by ChatGPT with some correction
    # CAUTION: THIS FUNCTION HAS NOT BEEN UNIT-TESTED.

    weights = (x - voxel_min_vertex) / (voxel_max_vertex - voxel_min_vertex)  # B x 4

    # step 1
    # 0->0000, 1->0001, 2->0010, 3->0011, 4->0100, 5->0101, 6->0110, 7->0111
    # 8->1000, 9->1001,10->1010,11->1011,12->1100,13->1101,14->1110,15->1111
    c000 = voxel_embedds[:, 0] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 8] * weights[:, 0][:, jnp.newaxis]
    c001 = voxel_embedds[:, 1] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 9] * weights[:, 0][:, jnp.newaxis]
    c010 = voxel_embedds[:, 2] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 10] * weights[:, 0][:, jnp.newaxis]
    c011 = voxel_embedds[:, 3] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 11] * weights[:, 0][:, jnp.newaxis]
    c100 = voxel_embedds[:, 4] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 12] * weights[:, 0][:, jnp.newaxis]
    c101 = voxel_embedds[:, 5] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 13] * weights[:, 0][:, jnp.newaxis]
    c110 = voxel_embedds[:, 6] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 14] * weights[:, 0][:, jnp.newaxis]
    c111 = voxel_embedds[:, 7] * (1 - weights[:, 0][:, jnp.newaxis]) + voxel_embedds[:, 15] * weights[:, 0][:, jnp.newaxis]

    # step 2
    c00 = c000 * (1 - weights[:, 1][:, jnp.newaxis]) + c100 * weights[:, 1][:, jnp.newaxis]
    c01 = c001 * (1 - weights[:, 1][:, jnp.newaxis]) + c101 * weights[:, 1][:, jnp.newaxis]
    c10 = c010 * (1 - weights[:, 1][:, jnp.newaxis]) + c110 * weights[:, 1][:, jnp.newaxis]
    c11 = c011 * (1 - weights[:, 1][:, jnp.newaxis]) + c111 * weights[:, 1][:, jnp.newaxis]

    # step 3
    c0 = c00 * (1 - weights[:, 2][:, jnp.newaxis]) + c10 * weights[:, 2][:, jnp.newaxis]
    c1 = c01 * (1 - weights[:, 2][:, jnp.newaxis]) + c11 * weights[:, 2][:, jnp.newaxis]

    # step 4
    c = c0 * (1 - weights[:, 3][:, jnp.newaxis]) + c1 * weights[:, 3][:, jnp.newaxis]

    return c
