import jax
import numpy as np
from flax.optim import OptimizerDef

def initialize_parameters(model, input_example, random_seed=42):
    """
    takes a FLAX model, an example input tensor and a random seed (to insure reproducibility)
    produces initial parameters for the model
    """
    rng = jax.random.PRNGKey(random_seed)
    parameters = model.init(rng, *input_example)
    return parameters

def training_loop(experiment,
                  parameters, loss_function, optimizer,
                  train_dataset_source, test_dataset,
                  num_epochs, batch_size,
                  display=True, random_seed=42):
    """
    `experiment` contains the names of the eements being tested and will be updated with the losses and other informations
    `parameters` is the initial parameters for the model that needs to be optimized
    `loss_function` takes parameters, a batch and returns the loss
    `optimizer` is a FLAX optimizer description
    `train_dataset_source` is a datasources that has a `batch_iterator` method to get a batch
    `test_dataset` is a batch
    `num_epochs` is the number of epochs for which the optimizer will run
    `random_seed` is a seed used for reproducibility
    """
    # initialization
    if isinstance(optimizer, OptimizerDef): optimizer = optimizer.create(parameters)
    np.random.seed(random_seed) # insures reproducible batch order

    # jitted loss function
    @jax.jit
    def train_step(optimizer, batch):
        def loss_fn(parameters): return loss_function(parameters, batch)
        loss_value, loss_grad = jax.value_and_grad(loss_fn)(optimizer.target)
        optimizer = optimizer.apply_gradient(loss_grad)
        return optimizer, loss_value
    loss_jitted = jax.jit(loss_function)

    # training loop
    if display: print("Starting training...")
    for _ in range(num_epochs):
        experiment.start_epoch()
        # iterates on all batches
        batch_iterator = train_dataset_source.batch_iterator(batch_size=batch_size, shuffle=True)
        for batch in batch_iterator:
            optimizer, loss_value = train_step(optimizer, batch)
            experiment.end_iteration(train_loss=loss_value, learning_rate=optimizer.learning_rate)
        # measure validation error
        test_loss = loss_jitted(optimizer.target, test_dataset)
        experiment.end_epoch(test_loss, display=display)

    if display: print(f"Training done (final test accuracy: {experiment.best_test_loss:e}).")
    return experiment
