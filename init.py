import jax.numpy as jnp
from jax import random

def init_eig_magnitude(r_min=0., r_max=1.):
    
    def init(key, shape):
        u = random.uniform(key=key, shape=shape)
        nu_log = jnp.log(-0.5 * jnp.log(u * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        return nu_log
    
    return init

def init_eig_phase(max_phase):

    def init(key, shape):
        u = random.uniform(key=key, shape=shape)
        theta_log = jnp.log(u * max_phase)
        return theta_log
    
    return init

def init_gamma_log(key, diag_lambda):
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))
