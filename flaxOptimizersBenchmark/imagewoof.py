import tensorflow_datasets as tfds
import tensorflow as tf
from .architectures import ResNet50
from .training_loop import cross_entropy, accuracy, training_loop, make_training_loop_description, make_problem_description, Experiment
from .dataset_builders.imagewoof import Imagewoof

# defines the model
model = ResNet50(num_classes=10)
model_name = "ResNet50"

# defines the training parameters
batch_size = 64
nb_epochs = 5
pictureSize = 128

# defines the version of imagewoof that will be used
imagewooffull = 'full-size-v2', 'imagewoofFullSize'
imagewoof320 = '320px-v2', 'imagewoof320'
imagewoof160 = '160px-v2', 'imagewoof160'
imagewoof_version, imagewoof_name = imagewoof160

def load_imagewoof(dataset_path):
    """
    gets the dataset from the given path
    this function works ONLY if the dataset has been previously downloaded
    """
    # this download by default, see https://www.tensorflow.org/datasets/api_docs/python/tfds/load
    builder = Imagewoof(config=imagewoof_version, data_dir=dataset_path)
    builder.download_and_prepare()
    train_dataset, test_dataset = builder.as_dataset(split=['train','validation'], as_supervised=True, shuffle_files=True)
    # converts values to floats between 0.0 and 1.0
    def normalize_picture(inputs,labels):
        inputs = tf.image.resize(inputs, size=(pictureSize, pictureSize), antialias=True)
        return float(inputs) / 255.0, labels
    train_dataset = train_dataset.map(normalize_picture, deterministic=False)
    test_dataset = test_dataset.map(normalize_picture, deterministic=False)
    return train_dataset, test_dataset

def run_imagewoof(dataset_path, optimizer_with_description, test_metrics={"accuracy":accuracy}, output_folder="../data", random_seed=None, display=True):
    """
    Runs a SimpleCNN model on the imagewoof dataset

    `optimizer_with_description` is an optimizer and its description as produced by `make_optimizer`
    `test_metrics` is a lost of function that will be evaluated on the test dataset at the end of each epoch
    `output_folder` is the folder where the result of the experiment will be stored
    `random_seed` can be specified to make the experiment deterministic
    `display` can be set to false to hide training informations
    """
    # description of the problem
    optimizer, optimizer_description, optimizer_metrics = optimizer_with_description
    training_loop_description = make_training_loop_description(nb_epochs=nb_epochs, batch_size=batch_size, random_seed=random_seed)
    problem_description = make_problem_description(benchmark_name=imagewoof_name, model_name=model_name, training_loop_description=training_loop_description, optimizer_description=optimizer_description)
    experiment = Experiment(problem_description, output_folder)
    # trains the model
    train_dataset, test_dataset = load_imagewoof(dataset_path)
    return training_loop(experiment, model, cross_entropy, optimizer, optimizer_metrics, test_metrics, train_dataset, test_dataset, display)
