import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import argparse

from models import MLP_Classifier
from experiments import Logger, initiate_run, load_config

from models import train_model
from utils import *

def run_task_1_train(run_name: str):

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # Initialize run
    exp_name = 'task_1_svhn_classifier'
    if initiate_run(exp_name=exp_name, run_name=run_name) != 2:
        print('Run newly initialized. Config file might be faulty if not correctly initialized.') 
        
    # Load the configuration
    config = load_config(exp_name, run_name)

    # Set random seed
    set_random_seed(config['random_seed'])
    # Load the dataset
    if config['dataset'] =='svhn':
        train_dataset, test_dataset = load_svhn_dataset()
    
    elif config['dataset'] == 'mnist':
        train_dataset, test_dataset = load_mnist_dataset()


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['eval_batch_size'], shuffle=False)

    # Initialize the model according to the config file
    mlp_classifier = MLP_Classifier(
                    input_dim=config['input_dim'],
                    mlp_layers=config['layers'],
    )
    mlp_classifier.to(device)
    # Define Criterion for the loss function
    if config['loss'] == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()

    # Define the optimizer
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(mlp_classifier.parameters(), lr=config['optimizer']['learning_rate'], betas=tuple(config['optimizer']['betas']), eps=config['optimizer']['eps'])

    # Initialize the logger
    logger = Logger(
                    exp_name=exp_name, 
                    run_name=run_name, 
                    configuration=config,
                    verbose=True,
                    log_gradients=False,
                    log_data = True,
                    checkpoint_frequency=config['save_frequency']
    )

    train_model(
                    model=mlp_classifier, 
                    train_dataset=train_loader, 
                    eval_dataset=test_loader, 
                    optimizer=optimizer,
                    criterion=criterion,
                    config=config,
                    logger=logger,
    )




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-task', type=int, required=True)
    parser.add_argument('-run', type=str, help="Run name")
    args = parser.parse_args()

    if args.task == 0:
        print('Task 0: Testing')
    if args.task == 1:
        print('Task 1: Training')
        run_task_1_train(args.run)
