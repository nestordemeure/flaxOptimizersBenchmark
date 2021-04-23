
#----------------------------------------------------------------------------------------
# METRICS

def get_hyperparams_lr(optimizer_def):
    """
    returns the learning rate stored in the hyperparameters
    """
    return optimizer_def.hyper_params.learning_rate

def get_hyperparams_wd(optimizer_def):
    """
    returns the weight decay stored in the hyperparameters
    """
    return optimizer_def.hyper_params.weight_decay

#----------------------------------------------------------------------------------------
# BUILDER

def make_optimizer(opt_name, opt_class, opt_metrics={'lr':get_hyperparams_lr}, **opt_initial_parameters):
    """
    takes a name, a class, some optional metrics (optimizer_def->float) and initial parameters
    returns an initialized optimizer, a description to be used inside experiment and metrics
    as a pair (optimizer,description,metrics)
    """
    # initialize optimizer
    optimizer = opt_class(**opt_initial_parameters)
    # creates description
    description = opt_initial_parameters
    description["name"] = opt_name
    return optimizer, description, opt_metrics
