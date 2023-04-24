import torch
import torch.nn as nn

from .training import run_train_epoch
from .evaluation import run_evaluation
from .tuning import HyperOpt
from experiments import Logger

#####---- General Execution Scripts ----#####

## These scripts can be re-used to train, evaluate or optimize different models quick. ##

## To log training progress, any evaluation metric or optimization run, simply use the logger. ##


###--- Training Scripts ---###

## Run simple training ##
def train_model_01(model: nn.Module, 
                train_dataset: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Optimizer, 
                criterion: nn.Module, 
                config: dict, 
                eval_dataset: torch.utils.data.DataLoader = None, 
                logger: Logger = None,
                verbose: bool = True):
    """
    Train the model on the provided dataset.

    Parameters:
        model (nn.Module): Model to train.
        train_dataset (torch.utils.data.DataLoader): Dataset to train the model on.
        eval_dataset (torch.utils.data.DataLoader): Dataset to evaluate the model on.
        config (dict): Dictionary containing the configuration of the model.
    """


    ###----Hyperparameters----###
    RUN_NAME = config["run_name"]
    EPOCHS = config["num_epochs"]

    EVAL_FREQUENCY = config["eval_frequency"]
 
    VERBOSE = verbose

    ###----Training----###

    # Display
    output = ''
    if verbose:
        output = f"----Training for model {RUN_NAME}----\n\nHyperparameters:\n "
        for param in config:
            output += f"  {param}: {config[param]}\n"
    output += "Start training...\n"
    print(output)
    output = ''

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train the model

    for epoch in range(EPOCHS):
        losses = data = eval_metrics = None
        
        print(f'Epoch {epoch + 1}/{EPOCHS}----------')
        accuracy = -1

        output += f'Epoch {epoch+1} Results:\n'

        # Train for one epoch   
        losses = run_train_epoch(model=model, dataset=train_dataset, config=config, criterion=criterion, optimizer=optimizer, device=device)
        output += f'  Train Loss: {sum(losses)/len(losses)}\n'
        # Evaluate on validation set.
        if (epoch+1) % EVAL_FREQUENCY == 0 and eval_dataset is not None:
            eval_metrics = run_evaluation(model=model, dataset=eval_dataset, config=config)
            for k, v in eval_metrics.items():
                output += f'  {k}: {v}\n'
        
        if verbose:
            print(output)

        output = ''

        if logger is not None:
            logger.step(epoch=epoch, model=model, optimizer=optimizer)
            data = {**{'train_loss': sum(losses)/len(losses)}, **eval_metrics} if eval_metrics is not None else {'train_loss': sum(losses)/len(losses)}
            logger.log(data=data)
        
    import ipdb; ipdb.set_trace()
    if VERBOSE:
        print(f'-----Training Complete. Final results-----\n')
        print(f'  Train Loss: {sum(losses)/len(losses)}')

###--- Evaluation Scripts ---###
 
def evaluate_model_01(model: nn.Module, 
                       eval_dataset: torch.utils.data.DataLoader, 
                       config: dict, 
                       device: str = 'cpu',
                       epoch: int = None,
                       verbose: bool = True, 
                       logger: Logger = None):
                       
    eval_metrics = run_evaluation(model=model, dataset=eval_dataset, config=config, device=device, logger=logger, verbose=verbose)
    logger.step(epoch=epoch+1, data=eval_metrics)
    if verbose:
        print(f'-----Evaluation Complete. Final results-----\n')
        # Print out the directory of evaluation metrics
        print(f' Evaluation Metric:\n')
        print('\n'.join([f'    {key}: {value}' for key, value in eval_metrics.items()]))


###--- Optimization Scripts ---###

def optimize_model_01(
                      config: dict,
                      optimization_config: list,
                      verbose: bool = True, 
                      logger: Logger = None):
    """
        Train and evalate the model on the provided dataset for hyperparameter tuning.

        Parameters:
            model (nn.Module): Model to train.
            train_dataset (torch.utils.data.DataLoader): Dataset to train the model on.
            eval_dataset (torch.utils.data.DataLoader): Dataset to evaluate the model on.
            config (dict): Dictionary containing the configuration of the model.
    """

    # Set up hyperparameter tuner
    tuner = HyperOpt(optimization_config=optimization_config, log=True)
    tuner._link_logger(logger=logger)
    tuner.tune_params(config = config)

        
