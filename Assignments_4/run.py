import argparse
import os

import torch

import utils

import torchgadgets as tg


###--- Training ---###

def training(exp_name, run_name):
    ##-- Load Config --##
    load_augm_config_train = utils.load_config('augm_train_preLoad')
    load_augm_config_test = utils.load_config('augm_test_preLoad')
    config = utils.load_config_from_run(exp_name, run_name)
    ##-- Load Dataset --##
    data = tg.data.load_dataset('oxfordpets')
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_dataset = tg.data.ImageDataset(dataset=train_dataset, transforms=load_augm_config_train)
    test_dataset = tg.data.ImageDataset(dataset=test_dataset, transforms=load_augm_config_test, train_set=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, drop_last=True)
    ##-- Logging --##
    log_dir = os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs')
    checkpoint_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints')
    logger = tg.logging.Logger(log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_config=config)
    
    tg.training.trainNN(config=config, logger=logger, train_loader=train_loader, test_loader=test_loader, return_all=False)



###--- Evaluation ---###

def evaluation(exp_name, run_name):
    print("Evaluation not implemented...")
    return


###--- Hyperparameter Tuning ---###

def hyperparameter_tuning(exp_name, run_name):
    print("Hyperparameter tuning not implemented...")
    return




if __name__ == '__main__':
    import ipdb; ipdb.set_trace()

    argparser = argparse.ArgumentParser()

    # Flags to signal which script to run
    argparser.add_argument('--train', action='store_true', default=False, help='Train the model')
    argparser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate the model')
    argparser.add_argument('--tuning', action='store_true', default=False, help='Tune the hyperparameters')

    argparser.add_argument('--init_exp', action='store_true', default=False, help='Initialize a new experiment')
    argparser.add_argument('--init_run', action='store_true', default=False, help='Initialize a new run')

    argparser.add_argument('--copy_conf', action='store_true', default=False, help='Load a configuration file to run')

    # Hyperparameters 
    argparser.add_argument('-exp', type=str, default=None, help='Experiment name')
    argparser.add_argument('-run', type=str, default=None, help='Run name')
    argparser.add_argument('-conf', type=str, default=None, help='Config name')

    args = argparser.parse_args()
    # If no experiment or run name is provided, the environment variables defining these have to be set
    if args.init_exp:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ), 'Please provide an experiment name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        utils.create_experiment(exp_name)
    
    if args.init_run:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        utils.create_run(exp_name, run_name)
    
    if args.copy_conf:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ) and args.conf is not None, 'Please provide an experiment and run name and the name of the config file'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        config_name = args.conf if args.conf is not None else os.environ.get('CURRENT_CONFIG')
        utils.load_config(exp_name, run_name, config_name)

    if args.tuning:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        hyperparameter_tuning(exp_name, run_name)

    if args.train:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        training(exp_name, run_name)

    if args.evaluate:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        evaluation(exp_name, run_name)
    

