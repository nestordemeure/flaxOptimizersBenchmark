import random
import tensorflow_datasets as tfds
from .architectures import ResNet18
from .training_loop import cross_entropy, training_loop, Experiment

# TODO this download by default, see https://www.tensorflow.org/datasets/api_docs/python/tfds/load
(train_dataset, test_dataset), info = tfds.load('mnist', split=['train','test'], as_supervised=True, shuffle_files=True, with_info=True)
nb_classes = info.features['label'].num_classes

# defines the model
model=ResNet18(num_classes=nb_classes)
model_name="ResNet18"

# defines the training parameters
batch_size=256
num_epochs=10

def run_MNIST(optimizer, optimizer_name, output_folder="../data", random_seed=None, display=True):
    """
    Runs a ResNet18 model on the MNIST dataset
    """
    if random_seed is None: random_seed = random.getrandbits(32)
    # defines the loss
    def loss(parameters, batch, train, use_mean=True):
        inputs, targets = batch
        predictions, updated_state = model.apply(parameters, x=inputs, train=train, mutable=['batch_stats'])
        return cross_entropy(predictions, targets, use_mean), updated_state
    # trains the model
    experiment = Experiment("MNIST", model_name, optimizer_name, output_folder)
    return training_loop(experiment, model, loss, optimizer, train_dataset, test_dataset, num_epochs, batch_size, display, random_seed)
