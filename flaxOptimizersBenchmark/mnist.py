import random
import numpy as np
from batchup.datasets import mnist
from batchup.data_source import ArrayDataSource
from .architectures import ResNet18
from .training_loop import cross_entropy, initialize_parameters, training_loop, Experiment

# gets the dataset, downloads it if needed
dataset = mnist.MNIST(n_val=None)
# work aroud an error due to dataset not being encoded as numpy arrays
train_X = np.array(dataset.train_X, copy=False)
train_y = np.array(dataset.train_y, copy=False)
test_X = np.array(dataset.test_X, copy=False)
test_y = np.array(dataset.test_y, copy=False)
# gets dataset ready
train_dataset_source = ArrayDataSource([train_X, train_y])
test_dataset = (test_X, test_y)
nb_classes = 1 + max(test_y)

# defines the model
model=ResNet18(num_classes=nb_classes)
model_name="ResNet18"

# defines the training parameters
batch_size=512
num_epochs=10

def run_MNIST(optimizer, optimizer_name, output_folder="../data", random_seed=None, display=True):
    """
    Runs a ResNet18 model on the MNIST dataset
    """
    if random_seed is None: random_seed = random.getrandbits(32)
    # defines the loss
    def loss(parameters, batch, train):
        inputs, targets = batch
        predictions, updated_state = model.apply(parameters, x=inputs, train=train, mutable=['batch_stats'])
        return cross_entropy(predictions, targets), updated_state
    # trains the model
    experiment = Experiment("MNIST", model_name, optimizer_name, output_folder)
    parameters = initialize_parameters(model=model, input_example=test_X, random_seed=random_seed)
    return training_loop(experiment, parameters, loss, optimizer, train_dataset_source, test_dataset, num_epochs, batch_size, display, random_seed)
