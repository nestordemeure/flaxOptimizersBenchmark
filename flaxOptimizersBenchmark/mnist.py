import random
import tensorflow_datasets as tfds
from .architectures import SimpleCNN
from .training_loop import cross_entropy, training_loop, Experiment

# this download by default, see https://www.tensorflow.org/datasets/api_docs/python/tfds/load
(train_dataset, test_dataset), info = tfds.load('mnist', split=['train','test'], as_supervised=True, shuffle_files=True, with_info=True)
nb_classes = info.features['label'].num_classes
# converts values to floats between 0.0 and 1.0
def normalize_picture(inputs,labels): return float(inputs) / 255.0, labels
train_dataset = train_dataset.map(normalize_picture, deterministic=False)
test_dataset = test_dataset.map(normalize_picture, deterministic=False)

# defines the model
model=SimpleCNN(num_classes=nb_classes)
model_name="SimpleCNN"

# defines the training parameters
batch_size=256
num_epochs=10

def run_MNIST(optimizer, optimizer_name, output_folder="../data", random_seed=None, display=True):
    """
    Runs a SimpleCNN model on the MNIST dataset
    """
    if random_seed is None: random_seed = random.getrandbits(32)
    # defines the loss
    def loss(parameters, batch, train, use_mean=True):
        inputs, targets = batch
        predictions, updated_state = model.apply(parameters, x=inputs, train=train, mutable=['batch_stats'])
        return cross_entropy(predictions, targets, use_mean=use_mean), updated_state
    # trains the model
    experiment = Experiment("MNIST", model_name, optimizer_name, output_folder)
    return training_loop(experiment, model, loss, optimizer, train_dataset, test_dataset, num_epochs, batch_size, display, random_seed)
