from flax import optim
from flaxOptimizersBenchmark import run_SVHN

# optimizer
learning_rate = 1e-3
weight_decay = 1e-3
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
optimizer = optim.Adam(learning_rate=learning_rate, weight_decay=weight_decay, beta1=beta1, beta2=beta2, eps=eps)

# run benchmark
run_SVHN(optimizer, 'Adam')
