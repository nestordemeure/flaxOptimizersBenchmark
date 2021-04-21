from flax import optim
from flaxOptimizersBenchmark import run_MNIST

# optimizer
learning_rate = 1e-3
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.99
eps = 1e-8
optimizer = optim.Adam(learning_rate=learning_rate, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps)

# run benchmark
run_MNIST(optimizer, 'Adam')

# TODO very slow (more than 10s per epoch)
# normal ?
# overfits masively

# reproduce this
# https://www.tensorflow.org/tutorials/quickstart/advanced?hl=en