from flax import optim
from flaxOptimizersBenchmark import make_optimizer, load_mnist, run_MNIST

# downloads the dataset if needed
data_folder = "~/tensorflow_datasets"
load_mnist(data_folder)

# optimizer definition
optimizer = make_optimizer("Adam", optim.Adam,
                           learning_rate = 1e-3, weight_decay = 0.1, beta1 = 0.9, beta2 = 0.99, eps = 1e-8)

# run benchmark
run_MNIST(data_folder, optimizer)
