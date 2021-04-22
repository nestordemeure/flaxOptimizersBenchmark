# testing experiment saving and loading
from flaxOptimizersBenchmark import Experiment

# creates an experiment
experiment = Experiment('bench', 'model', 'opt', '../data')

# runs the experiment for a number of epochs and iterations
# with a dummy loss
for epoch in range(5):
    experiment.start_epoch()
    for i in range(5):
        train_loss_i = 0.1 * (epoch*5 + i)
        lr_i = 1e-3 / (epoch*5 + i + 1)
        experiment.end_iteration(train_loss=train_loss_i, learning_rate=lr_i)
    test_loss = 0.2 * i + epoch
    experiment.end_epoch(test_loss=test_loss)

# reload the experiment
experiment2 = Experiment.load(experiment.filename)
print(experiment2.average_time_per_epoch)
