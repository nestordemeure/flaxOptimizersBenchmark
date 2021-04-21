# Flax Optimizers Benchmark

The goal of this repository is to provide users with a collection of benchmark to evaluate Flax optimizers.

We aim to:
- give some fast benchmarks to quickly evaluate new optimizers
- give classic benchmarks and plotting functions to help authors of deeplearning papers that want to publish a new optimizer

## Installation

You can install this librarie with:

```
pip install git+https://github.com/nestordemeure/flaxOptimizersBenchmark.git
```

## Usage

### Fast benchmarks

**TODO**

### Slow benchmarks

**TODO**

## TODO

- struct to store hyperparameters or just one file per dataset (simpler at first)
- code to download datasets
- main training loop
- architectures
- way to repeat experiments

- add accuracy when it is meaningful

- plot
  the plotting function take experiments as inputs or (dataset,architecture) to compare optimizers
  - bar plot or jit/run time
  - bar plot of final train/test loss
  - plot of loss accross time (train and/or test) (one or all optimizers)
  - final loss as a function of starting lr

## Functionalities we *do not* have

- data augmentation (as it is very problem specific and we are focussing on the optimizer rather than the individual problems)
- learning rate and weight decay scheduler (might be added later)

## Flax optimizers

- [flaxOptimizers](https://github.com/nestordemeure/flaxOptimizers) contains implementations of a large number of optimizers in Flax.
- [AdahessianJax](https://github.com/nestordemeure/AdaHessianJax) contains my implementation of the Adahessian second order optimizer in Flax.
- [Flax.optim](https://github.com/google/flax/tree/master/flax/optim) contains the official Flax optimizers.

