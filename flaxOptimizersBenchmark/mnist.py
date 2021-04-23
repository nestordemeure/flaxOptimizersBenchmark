import tensorflow_datasets as tfds
from .architectures import SimpleCNN
from .training_loop import cross_entropy, accuracy, training_loop, make_training_loop_description, make_problem_description, Experiment

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
nb_epochs=10

def run_MNIST(optimizer_with_description, test_metrics={"accuracy":accuracy}, output_folder="../data", random_seed=None, display=True):
    """
    Runs a SimpleCNN model on the MNIST dataset

    `optimizer_with_description` is an optimizer and it sdescription as produced by `make_optimizer`
    `test_metrics` is a lost of function that will be evaluated on the test dataset at the end of each epoch
    `output_folder` is the folder where the result of the experiment will be stored
    `random_seed` can be specified to make the experiment deterministic
    `display` can be set to false to hide training informations
    """
    # description of the problem
    optimizer, optimizer_description = optimizer_with_description
    training_loop_description = make_training_loop_description(nb_epochs=nb_epochs, batch_size=batch_size, random_seed=random_seed)
    problem_description = make_problem_description(benchmark_name="MNIST", model_name=model_name, training_loop_description=training_loop_description, optimizer_description=optimizer_description)
    experiment = Experiment(problem_description, output_folder)
    # trains the model
    return training_loop(experiment, model, cross_entropy, optimizer, test_metrics, train_dataset, test_dataset, display)
