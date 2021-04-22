import tensorflow_datasets as tfds
import numpy as np

batch_size=3
(train_dataset, test_dataset), info = tfds.load('mnist', split=['train','test'], as_supervised=True, shuffle_files=True, with_info=True)
nb_classes = info.features['label'].num_classes

batched_dataset = train_dataset.batch(batch_size).prefetch(1)
(input,label) = next(batched_dataset.as_numpy_iterator())
print("input", input.shape)
print("label", label.shape)
print("input", np.max(input))
