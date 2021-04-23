import random
import statistics
import math
import time
import json
from collections import defaultdict

#----------------------------------------------------------------------------------------
# PROBLEM DESCRIPTION

def make_training_loop_description(nb_epochs, batch_size, random_seed=None):
    """
    produces a description of the training loop
    """
    if random_seed is None: random_seed = random.getrandbits(32)
    return {"nb_epochs":nb_epochs, "batch_size":batch_size, "random_seed":random_seed}

def make_problem_description(benchmark_name, model_name, training_loop_description, optimizer_description):
    """
    produces a full description of problem being solved
    """
    return {"benchmark_name":benchmark_name, "model_name": model_name,
            "optimizer_description":optimizer_description,
            "training_loop_description":training_loop_description}

#----------------------------------------------------------------------------------------
# EXPERIMENT

class Experiment():
    "Used to store training data on an experiment."
    def __init__(self, problem_description, save_folder_path):
        self.problem_description = problem_description
        self.save_folder = save_folder_path
        # current time, used as an id to discriminate between identical experiments
        self.id = str(time.time())
        # once per iteration information
        self.iteration_metrics = defaultdict(list)
        # once per epoch information
        self.iteration_at_epoch = [] # iteration at which the epochs have hapened
        self.epoch_runtimes = [] # time taken to run each epoch
        self.epoch_metrics = defaultdict(list)
        # used for time recording
        self.__epoch_start_time= None

    #--------------------------------------------------------------------------
    # PROPERTIES

    @property
    def nb_epochs(self):
        "number of epochs so far"
        return len(self.epoch_runtimes)

    @property
    def nb_iterations(self):
        "number of iterations so far"
        return len(self.iteration_metrics['train_loss'])

    @property
    def average_time_per_epoch(self):
        "average run time, not counting the first epoch as it is impacted by the jit"
        if len(self.epoch_runtimes) < 2: return math.nan
        return statistics.mean(self.epoch_runtimes[1:])

    @property
    def best_train_loss(self):
        "returns the best training loss that was observed so far"
        return min(self.iteration_metrics['train_loss'])

    @property
    def best_test_loss(self):
        "returns the best testing loss that was observed so far"
        return min(self.epoch_metrics['test_loss'])

    @property
    def jit_time(self):
        "aproximate time spent jitting the code"
        if len(self.epoch_runtimes) < 2: return math.nan
        return self.epoch_runtimes[0] - self.average_time_per_epoch

    @property
    def filename(self):
        "returns the path to the file where the experiment will be stored"
        benchmark_name = self.problem_description['benchmark_name']
        model_name = self.problem_description['model_name']
        optimizer_name = self.problem_description['optimizer_description']['name']
        nb_epochs = self.problem_description['training_loop_description']['nb_epochs']
        batch_size = self.problem_description['training_loop_description']['batch_size']
        return self.save_folder + '/' + benchmark_name + '_' \
               + model_name + '_' + str(nb_epochs) + '_' + str(batch_size) + '_' \
               + optimizer_name + '_' + self.id + '.json'

    #--------------------------------------------------------------------------
    # ITERATION

    def end_iteration(self, train_loss, **metrics):
        """
        records the training loss and any additional metric stored in the corresponding dictionary
        we suppose that this function will be called at the end of each iteration
        """
        self.iteration_metrics['train_loss'].append(train_loss)
        for (name,value) in metrics.items():
            self.iteration_metrics[name].append(value)

    #--------------------------------------------------------------------------
    # EPOCH

    def start_epoch(self):
        """
        sets a timer to measure the time spent in computations
        """
        self.__epoch_start_time = time.time()

    def end_epoch(self, test_loss, display=True, **metrics):
        """
        stores:
        - the iteration number,
        - the test loss,
        - any additional metric stored in the corresponding dictionary,
        - the runtime for the epoch
        save the Experiment to file
        optionally displays some informations
        """
        if self.__epoch_start_time is None: raise AssertionError("You need to call `start_epoch` at the begining of the epoch!")
        # stores iteration
        self.iteration_at_epoch.append(self.nb_iterations - 1) # -1 for indexing purposes
        # updates runtime information
        runtime = time.time() - self.__epoch_start_time
        self.epoch_runtimes.append(runtime)
        # stores test loss
        self.epoch_metrics['test_loss'].append(test_loss)
        for (name,value) in metrics.items():
            self.epoch_metrics[name].append(value)
        # saves the experiment as a json file
        self.save()
        # displays some information
        if display:
            latest_train_loss = self.iteration_metrics['train_loss'][-1]
            output = f'Epoch {self.nb_epochs} in {runtime:0.2f}s. Test loss: {test_loss:e} Train loss: {latest_train_loss:e}'
            for (name,values) in self.epoch_metrics.items():
                if name != 'test_loss': output += f' Test {name}: {values[-1]}'
            print(output)

    #--------------------------------------------------------------------------
    # SERIALIZATION

    def save(self):
        "saves experiment to disk as a json file"
        with open(self.filename, 'w') as file:
            json.dump(self.__dict__, fp=file, indent=4)

    @staticmethod
    def load(path):
        "load an experiment, stored as a json, given the corresponding path"
        with open(path, 'r') as file:
            experiment = Experiment({}, '') # empty dummy
            experiment.__dict__ = json.load(fp=file)
            return experiment
