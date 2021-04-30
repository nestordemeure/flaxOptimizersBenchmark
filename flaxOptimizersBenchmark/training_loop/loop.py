from functools import partial
from collections import defaultdict
import jax
import numpy as np
import flax
from typing import Any

def asscalar(x):
    """
    turns a device array returns by JAX into a proper scalar
    """
    return np.asscalar(np.array(x))

@flax.struct.dataclass
class TrainState:
    """
    stores the current network parameters, mutable state and optimizer state
    uses flax.struct.dataclass to pass to tree_map and pmap
    """
    step: int
    optimizer: flax.optim.Optimizer
    model_state: Any

def initialize_parameters(model, batched_dataset, random_seed=42):
    """
    takes a FLAX model, an example input tensor and a random seed (to insure reproducibility)
    produces initial parameters for the model
    """
    rng = jax.random.PRNGKey(random_seed)
    input_example, _ = next(batched_dataset.as_numpy_iterator())
    variables = model.init(rng, input_example)
    model_state, params = variables.pop('params')
    return model_state, params

def compute_average_metrics(dataset, state, apply_model, metrics_functions):
    """
    Evaluates each metric, averaged over all elements of the dataset
    we suppose that the functions in metrics_functions are jitted and do not compute the average themselves
    """
    # rebuilds the variables
    params = state.optimizer.target
    variables = {'params': params, **state.model_state}
    # compute sthe metrics
    metrics = defaultdict(float)
    nb_elements = 0
    for batch in dataset.as_numpy_iterator():
        # applies the model to the inputs
        inputs, targets = batch
        predictions = apply_model(variables, inputs)
        # evaluates all losses
        for (name,function) in metrics_functions.items():
            metric_batch = function(predictions, targets)
            metrics[name] += asscalar(metric_batch)
        nb_elements += batch[0].shape[0]
    # turns all sums into averages
    for name in metrics.keys():
        metrics[name] /= nb_elements
    return metrics

def training_loop(experiment,
                  model, loss_function, optimizer,
                  optimizer_metrics, per_epoch_metrics,
                  train_dataset, test_dataset,
                  display=True):
    """
    `experiment` contains the names of the elements being tested and will be updated with the losses and other informations

    `model` is the model that should be applied to the inputs
    `loss_function` takes (predictions,targets) and returns a scalar
    `optimizer` is a FLAX optimizer description

    `train_dataset` is a Tensorflow.Dataset
    `test_dataset` is a Tensorflow.Dataset

    `display` tells us whether we should display intermediate informations during the training
    """
    # gets training parameters
    nb_epochs = experiment.problem_description["training_loop_description"]['nb_epochs']
    batch_size = experiment.problem_description["training_loop_description"]['batch_size']
    # cut datasets into batches
    train_dataset = train_dataset.batch(batch_size).prefetch(1)
    test_dataset = test_dataset.batch(batch_size).prefetch(1)
    # initialize parameters
    random_seed =  experiment.problem_description["training_loop_description"]['random_seed']
    np.random.seed(random_seed) # insures reproducible batch order
    model_state, params = initialize_parameters(model=model, batched_dataset=train_dataset, random_seed=random_seed)
    optimizer = optimizer.create(params)
    state = TrainState(step=0, optimizer=optimizer, model_state=model_state)

    # jitted training step
    @jax.jit
    def train_step(optimizer, state, batch):
        optimizer = state.optimizer
        # gets the loss and updated parameter
        def loss_fn(params):
            inputs, targets = batch
            variables = {'params': params, **state.model_state}
            predictions, updated_batch_stats = model.apply(variables, inputs, mutable=['batch_stats'])
            loss = loss_function(predictions, targets, use_mean=True)
            return loss, updated_batch_stats
        # computes the gradient
        (loss_value, updated_batch_stats), loss_grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
        # applies the optimization
        updated_optimizer = optimizer.apply_gradient(loss_grad)
        updated_state = state.replace(optimizer=updated_optimizer, model_state=updated_batch_stats)
        return optimizer, loss_value, updated_state
    # jitted model application
    @jax.jit
    def apply_model_jitted(variables, inputs):
        return model.apply(variables, x=inputs)
    # jits all metrics
    # TODO might be able to produce a single jitted function that computes all the metrics at once
    per_epoch_metrics['test_loss'] = loss_function
    for (name,function) in per_epoch_metrics.items():
        per_epoch_metrics[name] = jax.jit(partial(function, use_mean=False))

    # training loop
    if display: print("Starting training...")
    for _ in range(nb_epochs):
        experiment.start_epoch()
        # iterates on all batches
        for batch in train_dataset.as_numpy_iterator():
            # does one optimization step and updates the batchnorm state
            optimizer, loss_value, state = train_step(optimizer, state, batch)
            # collect optimizer metrics
            metrics = {'train_loss': asscalar(loss_value)}
            for (name,function) in optimizer_metrics.items():
                metrics[name] = function(optimizer.optimizer_def)
            experiment.end_iteration(**metrics)
        # computes and stores epoch metrics
        experiment.end_epoch_timer()
        test_metrics = compute_average_metrics(test_dataset, state, apply_model_jitted, per_epoch_metrics)
        experiment.end_epoch(**test_metrics, display=display)
    if display: print(f"Training done (final test loss: {experiment.best_test_loss:e}).")
    return experiment
