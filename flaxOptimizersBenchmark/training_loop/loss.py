import jax.numpy as jnp

def mean_squared_error(predictions, targets):
    return jnp.square(predictions - targets).mean()

def mean_absolute_error(predictions, targets):
    return jnp.abs(predictions - targets).mean()
