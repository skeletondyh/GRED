import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import lecun_normal, variance_scaling, normal
from init import init_eig_magnitude, init_eig_phase, init_gamma_log
from utils import binary_operator_diag

recurrent_param = ["B_re", "B_im", "nu_log", "theta_log", "gamma_log"]
no_decay_param = ["embedding", "scale", "bias"]

# From https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py#L78
full_atom_feature_dims = [119, 5, 12, 12, 10, 6, 6, 2, 2]
full_bond_feature_dims = [5, 6, 2]

class MLP(nn.Module):

    dim_h: int
    expand: int = 1
    drop_rate: float = 0.

    @nn.compact
    def __call__(self, inputs, training: bool = False):
        x = nn.LayerNorm()(inputs)
        x = nn.Dense(self.expand * self.dim_h)(x)
        x = nn.gelu(x)
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        x = nn.Dense(self.dim_h)(x)
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        return x + inputs

class LRU(nn.Module):

    dim_v: int
    dim_h: int

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"
    
    @nn.compact
    def __call__(self, inputs, training: bool = False):
        
        xs = nn.LayerNorm()(inputs)

        nu_log = self.param("nu_log", init_eig_magnitude(self.r_min, self.r_max), (self.dim_v,))
        theta_log = self.param("theta_log", init_eig_phase(self.max_phase), (self.dim_v,))
        diag_lambda = jnp.exp(-jnp.exp(nu_log) + 1j * jnp.exp(theta_log))
        gamma_log = self.param("gamma_log", init_gamma_log, diag_lambda)
        
        B_re = self.param("B_re", variance_scaling(0.5, "fan_in", "truncated_normal"), (self.dim_h, self.dim_v))
        B_im = self.param("B_im", variance_scaling(0.5, "fan_in", "truncated_normal"), (self.dim_h, self.dim_v))
        B = B_re + 1j * B_im
        Bu = xs @ (B * jnp.exp(gamma_log))

        lambda_elements = jnp.repeat(diag_lambda[None, ...], inputs.shape[0], axis=0)
        lambda_elements = jnp.expand_dims(lambda_elements, axis=(1, 2))
        _, xs = jax.lax.associative_scan(binary_operator_diag, (lambda_elements, Bu), reverse=True)
        x = xs[0]

        C_re = self.param("C_re", lecun_normal(), (self.dim_v, self.dim_h))
        C_im = self.param("C_im", lecun_normal(), (self.dim_v, self.dim_h))
        C = C_re + 1j * C_im

        x = nn.gelu((x @ C).real)
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        if self.act == "full-glu":
            x = nn.Dense(self.dim_h)(x) * jax.nn.sigmoid(nn.Dense(self.dim_h)(x))
        elif self.act == "half-glu":
            x = x * jax.nn.sigmoid(nn.Dense(self.dim_h)(x))
        x = nn.Dropout(self.drop_rate, deterministic=not training)(x)
        return x + inputs[0]
    
class GRED(nn.Module):

    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, dist_masks, training: bool = False):
        xs = jnp.swapaxes(dist_masks, 0, 1) @ inputs
        xs = MLP(self.dim_h, self.expand, self.drop_rate)(xs)
        x = LRU(
            self.dim_v,
            self.dim_h,
            self.r_min,
            self.r_max,
            self.max_phase,
            self.drop_rate,
            self.act
        )(xs, training=training)
        return x

class ZINC(nn.Module):

    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, node_masks, dist_masks, edge_attr = None, training: bool = False):
        x = nn.Embed(28, self.dim_h, embedding_init=normal(stddev=0.01))(inputs)
        x = nn.Dense(self.dim_h)(nn.gelu(x))
        e = nn.Embed(4, self.dim_h, embedding_init=normal(stddev=0.01))(edge_attr)
        e = nn.Dense(self.dim_h)(nn.gelu(e))
        deg = jnp.sum(dist_masks[:, 1], axis=-1, keepdims=True)
        deg_inv = jnp.where(deg > 0, 1 / deg, 0)
        x = x + jnp.sum(dist_masks[:, 1, ..., None] * e, axis=-2) * deg_inv
        
        for _ in range(self.num_layers):
            x = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x, dist_masks, training=training)
        x = jnp.where(jnp.expand_dims(node_masks, -1), x, 0.)
        x = jnp.sum(x, axis=1)
        x = nn.gelu(nn.Dense(self.dim_h)(x))
        x = nn.Dense(self.dim_o)(x)
        return x

class Peptides(nn.Module):

    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, node_masks, dist_masks, training: bool = False):
        x = 0
        for i in range(inputs.shape[-1]):
            x = x + nn.Embed(full_atom_feature_dims[i], self.dim_h, embedding_init=normal(stddev=0.01))(inputs[..., i])
        x = nn.Dense(self.dim_h)(nn.gelu(x))

        for _ in range(self.num_layers):
            x = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x, dist_masks, training=training)
        x = jnp.where(jnp.expand_dims(node_masks, -1), x, 0.)
        x = jnp.sum(x, axis=1)
        x = nn.gelu(nn.Dense(self.dim_h)(x))
        x = nn.Dense(self.dim_o)(x)
        return x

class SuperPixel(nn.Module):

    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, node_masks, dist_masks, training: bool = False):
        x = nn.Dense(self.dim_h)(inputs)
        x = nn.Dense(self.dim_h)(nn.gelu(x))
        for _ in range(self.num_layers):
            x = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x, dist_masks, training=training)
        x = jnp.where(jnp.expand_dims(node_masks, -1), x, 0.)
        x = jnp.sum(x, axis=1) / jnp.sum(node_masks, axis=1, keepdims=True)
        x = nn.Dense(self.dim_o)(x)
        return x

class SBM(nn.Module):

    num_layers: int
    dim_o: int

    dim_v: int
    dim_h: int
    expand: int = 1

    r_min: float = 0.
    r_max: float = 1.
    max_phase: float = 6.28
    drop_rate: float = 0.
    act: str = "full-glu"

    @nn.compact
    def __call__(self, inputs, dist_masks, training: bool = False):
        x = nn.Embed(7, self.dim_h, embedding_init=normal(stddev=0.01))(inputs.argmax(axis=-1))
        for _ in range(self.num_layers):
            x = GRED(
                self.dim_v,
                self.dim_h,
                self.expand,
                self.r_min,
                self.r_max,
                self.max_phase,
                self.drop_rate,
                self.act
            )(x, dist_masks, training=training)
        x = nn.Dense(self.dim_o)(x)
        return x
