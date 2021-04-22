# Flax Optimizers Benchmark

The goal of this repository is to provide users with a collection of benchmark to evaluate Flax optimizers.
To do so, we buil an infrastructure to run optimizers on a panel of datasets (adapted from [TensorFlow Datasets](https://www.tensorflow.org/datasets/overview)), save all meaningful training metrics to disk and plot their results later.

We aim to:
- give some fast benchmarks to quickly evaluate new optimizers
- give classic benchmarks, plotting functions and functionalities to help authors of deep-learning papers that want to publish a new optimizer

**This is a work in progress, you should expect the elements in the TODO list to be done within a few weeks.**

## Installation

You can install this librarie with:

```
pip install git+https://github.com/nestordemeure/flaxOptimizersBenchmark.git
```

## Usage

**TODO**

## TODO

- provide code to download and load datasets separately and to a user defined folder

- the `Experiment` object should store additional information
  - optimizer initial parameters
  - training loop initial parameters
  - metrics computed during the run (losses but also accuracy where it is meaningful)

- code to repeat experiments

- add some datasets
  - imagenette
  - imagewoof
  - imagenet
  - COCO
  - wikitext

- plotting functions that take `Experiment` as input (or (dataset,architecture) to compare optimizers)
  - bar plot of jit/run time
  - bar plot of final train/test loss
  - plot of metrics accross time (train and/or test) (one or all optimizers)
  - final loss as a function of starting lr

## Functionalities we *do not* have

- data augmentation (as it is very problem specific and we are focussing on the optimizer rather than the individual problems)
- learning rate and weight decay scheduler (might be added later)

## Flax optimizers

You can find optimizers compatible with Flax in the following repositories:

- [flaxOptimizers](https://github.com/nestordemeure/flaxOptimizers) contains implementations of a large number of optimizers in Flax.
- [AdahessianJax](https://github.com/nestordemeure/AdaHessianJax) contains my implementation of the Adahessian second order optimizer in Flax.
- [Flax.optim](https://github.com/google/flax/tree/master/flax/optim) contains the official Flax optimizers.
