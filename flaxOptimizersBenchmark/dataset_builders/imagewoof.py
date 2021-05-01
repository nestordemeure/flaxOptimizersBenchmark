# adapted from the imagenette builder
# TODO: remove once imagewoof is integrated into tensorflow-datasets
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/imagenette.py
import os
import tensorflow as tf
import tensorflow_datasets.public_api as tfds

_CITATION = """
@misc{imagewoof,
  author    = "Jeremy Howard",
  title     = "imagewoof",
  url       = "https://github.com/fastai/imagenette/"
}
"""

_DESCRIPTION = """\
Imagewoof is a subset of 10 classes from Imagenet that aren't so easy to classify, since they're all dog breeds.
The breeds are: Australian terrier, Border terrier, Samoyed, Beagle, Shih-Tzu, English foxhound, Rhodesian ridgeback, Dingo, Golden retriever, Old English sheepdog.
(No we will not enter in to any discussion in to whether a dingo is in fact a dog. Any suggestions to the contrary are un-Australian.
Thank you for your cooperation.)
 * Full size download;
 * 320 px download;
 * 160 px download.
Note: The v2 config correspond to the new 70/30 train/valid split (released
in Dec 6 2019).
"""

#_LABELS_FNAME = "/image_classification/imagewoof_labels.txt"
_LABELS_FNAME = os.path.join(os.path.dirname(__file__), "imagewoof_labels.txt")
_URL_PREFIX = "https://s3.amazonaws.com/fast-ai-imageclas/"

class ImagewoofConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Imagewoof."""

  def __init__(self, size, base, **kwargs):
    super(ImagewoofConfig, self).__init__(
        # `320px-v2`,...
        name=size + ("-v2" if base == "imagewoof2" else ""),
        description="{} variant.".format(size),
        **kwargs)
    # e.g. `imagewoof2-320.tgz`
    self.dirname = base + {
        "full-size": "",
        "320px": "-320",
        "160px": "-160",
    }[size]

def _make_builder_configs():
  configs = []
  for base in ["imagewoof2", "imagewoof"]:
    for size in ["full-size", "320px", "160px"]:
      configs.append(ImagewoofConfig(base=base, size=size))
  return configs

class Imagewoof(tfds.core.GeneratorBasedBuilder):
  """A smaller subset of 10 easily classified classes from Imagenet."""

  VERSION = tfds.core.Version("1.0.0")

  BUILDER_CONFIGS = _make_builder_configs()

  def _info(self):
    names_file = tfds.core.tfds_path(_LABELS_FNAME)
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            "image": tfds.features.Image(encoding_format="jpeg"),
            "label": tfds.features.ClassLabel(names_file=names_file)
        }),
        supervised_keys=("image", "label"),
        homepage="https://github.com/fastai/imagenette",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    dirname = self.builder_config.dirname
    url = _URL_PREFIX + "{}.tgz".format(dirname)
    path = dl_manager.download_and_extract(url)
    train_path = os.path.join(path, dirname, "train")
    val_path = os.path.join(path, dirname, "val")

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "datapath": train_path,
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                "datapath": val_path,
            },
        ),
    ]

  def _generate_examples(self, datapath):
    """Yields examples."""
    for label in tf.io.gfile.listdir(datapath):
      for fpath in tf.io.gfile.glob(os.path.join(datapath, label, "*.JPEG")):
        fname = os.path.basename(fpath)
        record = {
            "image": fpath,
            "label": label,
        }
        yield fname, record