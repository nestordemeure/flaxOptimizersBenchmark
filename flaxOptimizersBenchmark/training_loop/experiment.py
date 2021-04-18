import statistics
import math
import time
import json

# TODO:
# - function to get best training and testing loss
# - function to get the mean and sd of several Experiments
# - pass dictionnary rather than particular loss/parameter so that we can instrument many things (?)

class Experiment():
    "Used to store training data on an experiment."
    def __init__(self, benchmark_name, model_name, optimizer_name, save_folder_path):
        self.benchmark_name = benchmark_name
        self.model_name = model_name
        self.optimizer_name = optimizer_name
        self.save_folder = save_folder_path
        # current time, used as an id to discriminate between identical experiments
        self.id = str(time.time())
        # per iteration information
        self.train_losses = [] # one per iteration
        self.learning_rates = [] # learning rate used at each iteration
        # per epoch information
        self.iteration_at_epoch = [] # iteration at which the epochs have hapened
        self.test_losses = []
        self.epoch_runtimes = [] # time taken to run each epoch
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
        return len(self.train_losses)

    @property
    def average_time_per_epoch(self):
        "average run time, not counting the first epoch as it is impacted by the jit"
        if len(self.epoch_runtimes) < 2: return math.nan
        return statistics.mean(self.epoch_runtimes[1:])

    @property
    def jit_time(self):
        "aproximate time spent jitting the code"
        if len(self.epoch_runtimes) < 2: return math.nan
        return self.epoch_runtimes[0] - self.average_time_per_epoch

    @property
    def filename(self):
        "returns the path to the file where the experiment will be stored"
        return self.save_folder + '/' + self.benchmark_name + '_' \
               + self.model_name + '_' + self.optimizer_name + '_' + self.id \
               + '.json'

    #--------------------------------------------------------------------------
    # ITERATION

    def end_iteration(self, train_loss, learning_rate=None):
        """
        stores training loss and, optionaly, learning rate
        we suppose that this function will be called at the end of each iteration
        """
        self.train_losses.append(train_loss)
        self.learning_rates.append(learning_rate)

    #--------------------------------------------------------------------------
    # EPOCH

    def start_epoch(self):
        """
        sets a timer to measure the time spent in computations
        """
        self.__epoch_start_time = time.time()

    def end_epoch(self, test_loss, display=True):
        """
        stores:
        - the iteration number,
        - the test loss,
        - the runtime for the epoch
        save the Experiment to file
        optionally displays some informations
        """
        if self.__epoch_start_time is None: raise AssertionError("You need to call `start_epoch` at the begining of the epoch!")
        # stores iteration
        self.iteration_at_epoch.append(self.nb_iterations - 1) # -1 for indexing purposes
        # stores test loss
        self.test_losses.append(test_loss)
        # updates runtime information
        runtime = time.time() - self.__epoch_start_time
        self.epoch_runtimes.append(runtime)
        # saves the experiment as a json file
        self.save()
        # displays some information
        if display:
            latest_test_loss = self.test_losses[-1]
            latest_train_loss = self.train_losses[-1]
            print(f'Epoch {self.nb_epochs} in {runtime:0.2f}s. Test loss: {latest_test_loss:e} Train loss: {latest_train_loss:e}')

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
            experiment = Experiment('', '', '', '') # empty dummy
            experiment.__dict__ = json.load(fp=file)
            return experiment
