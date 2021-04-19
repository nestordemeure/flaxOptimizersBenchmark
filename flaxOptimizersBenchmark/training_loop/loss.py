import jax.numpy as jnp

def mean_squared_error(predictions, targets):
    return jnp.square(predictions - targets).mean()

def mean_absolute_error(predictions, targets):
    return jnp.abs(predictions - targets).mean()

def cross_entropy(logits, targets, target_is_onehot=False):
    if target_is_onehot: targets = jnp.argmax(targets, axis=1)
    nll = jnp.take_along_axis(logits, jnp.expand_dims(targets, axis=1), axis=1)
    return -nll.mean()
