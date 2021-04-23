import jax.numpy as jnp

def mean_squared_error(predictions, targets, use_mean=True):
    squares = jnp.square(predictions - targets)
    return jnp.mean(squares) if use_mean else jnp.sum(squares)

def mean_absolute_error(predictions, targets, use_mean=True):
    diffs = jnp.abs(predictions - targets)
    return jnp.mean(diffs) if use_mean else jnp.sum(diffs)

def cross_entropy(logits, targets, target_is_onehot=False, use_mean=True):
    if target_is_onehot: targets = jnp.argmax(targets, axis=1)
    nll = jnp.take_along_axis(logits, jnp.expand_dims(targets, axis=1), axis=1)
    return -jnp.mean(nll) if use_mean else -jnp.sum(nll)

def accuracy(logits, targets, use_mean=True):
    predictions = jnp.argmax(logits, axis=1)
    is_correct = (predictions == targets)
    return jnp.mean(is_correct) if use_mean else jnp.sum(is_correct)
