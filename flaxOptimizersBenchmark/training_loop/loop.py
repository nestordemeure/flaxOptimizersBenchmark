from functools import partial
import jax
import numpy as np
from flax.optim import OptimizerDef

def asscalar(x):
    """
    turns a device array returns by JAX into a proper scalar
    """
    return np.asscalar(np.array(x))

def initialize_parameters(model, batched_dataset, random_seed=42):
    """
    takes a FLAX model, an example input tensor and a random seed (to insure reproducibility)
    produces initial parameters for the model
    """
    rng = jax.random.PRNGKey(random_seed)
    input_example, _ = next(batched_dataset.as_numpy_iterator())
    parameters = model.init(rng, input_example, train=False)
    return parameters

def compute_average_loss(dataset, parameters, loss_summed_jitted):
    """
    Evaluates the loss, averaged over all elements in a dataset
    WARNING: the loss must return the sum of the loss for all element in the batch and not the average
    """
    loss = 0.0
    nb_elements = 0
    for batch in dataset.as_numpy_iterator():
        loss_batch, _ = loss_summed_jitted(parameters, batch)
        loss += asscalar(loss_batch)
        nb_elements += batch[0].shape[0]
    return loss / nb_elements

def training_loop(experiment,
                  model, loss_function, optimizer,
                  train_dataset, test_dataset,
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
    np.random.seed(random_seed) # insures reproducible batch order
    # cut datasets into batches
    train_dataset = train_dataset.batch(batch_size).prefetch(1)
    test_dataset = test_dataset.batch(batch_size).prefetch(1)

    # initialize parameters
    parameters = initialize_parameters(model=model, batched_dataset=train_dataset, random_seed=random_seed)
    if isinstance(optimizer, OptimizerDef): optimizer = optimizer.create(parameters)

    # jitted loss functions
    @jax.jit
    def train_step(optimizer, batch):
        def loss_fn(parameters): return loss_function(parameters, batch, train=True, use_mean=True)
        (loss_value, updated_state), loss_grad = jax.value_and_grad(loss_fn, has_aux=True)(optimizer.target)
        optimizer = optimizer.apply_gradient(loss_grad)
        return optimizer, loss_value, updated_state
    loss_summed_jitted = jax.jit(partial(loss_function, train=False, use_mean=False))

    # training loop
    if display: print("Starting training...")
    for _ in range(num_epochs):
        experiment.start_epoch()
        # iterates on all batches
        for batch in train_dataset.as_numpy_iterator():
            optimizer, loss_value, updated_state = train_step(optimizer, batch)
            optimizer.replace(target={'params': optimizer.target['params'], 'batch_stats': updated_state})
            experiment.end_iteration(train_loss=asscalar(loss_value), learning_rate=optimizer.optimizer_def.hyper_params.learning_rate)
        # measure validation error
        test_loss = compute_average_loss(test_dataset, optimizer.target, loss_summed_jitted)
        experiment.end_epoch(test_loss, display=display)
    if display: print(f"Training done (final test accuracy: {experiment.best_test_loss:e}).")
    return experiment
