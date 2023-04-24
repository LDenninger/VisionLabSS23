import torch
import torch.nn as nn
import torch.utils.data as data_utils

from tqdm import tqdm

import optuna

import os
import copy

from pathlib import Path as P

from models import *
from .utils import *
from .evaluation import *
from .utils import *
from experiments import Logger
from utils import *

class HyperOpt():

    def __init__(self, model_config: dict = None,
                         optimization_config: dict = None,
                            log: bool = False):
        
        """
        Hyperparameter tuner using optuna.

        Parameters:
            optimization_config (dict): A dictionary containing the configuration of the trial.
                Format:
                            [{  'parameter':  [
                                                {
                                                    'param': name of the parameter in the config file,
                                                    'type': parameter type in [categorical, dsicrete_uniform, float, int, loguniform, uniform],
                                                    'range': range [low, high] if continuous, discrete set [item_1,..., item_n] else
                                                }, ...
                                            ],
                                'n_trials' (int): Number of trials for the optimization of the parameters.
                                'eval_metric' (str): The evaluation metric to optimize.
                                'sampler' (str): The sampler to use. Possibble Sampler: ["TPESampler", "RandomSampler", "CmaEsSampler", "PartialFixedSampler", "NSGAIISampler", "MOTPESampler", "QMCSampler", "BruteForceSampler"]
                                'pruner; (str): The pruner to use. Possibble Pruner: ["MedianPruner, NopPruner, PatientPruner,, PercentilePruner, SuccessiveHalvingPruner, HyperbandPruner, ThresholdPruner]
                                'maximize' (bool): Whether to maximize or minimize the evaluation metric.}
                            ]
        """
        self.log = log

        self.logger = None

        self.train_dataset = None
        self.eval_dataset = None

        self.config = model_config
        self.optimization_config = [optimization_config] if optimization_config is not None else []


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tune_params(self,
                    config: dict
                    ):
        """
            Optimize the model hyper parameters by maximizing/minimizing the evaluation metric.
            Different Trials can be added using the add_trial() function prior to parameter tuning.
            The trials are then sequentially evaluated, such that each trial takes the optimized configuration from the previous trial.

            Arguments:
                config (dict): A dictionary containing the base configuration of the model.
        
        """
        assert len(self.optimization_config) > 0, 'No trial configuration provided.'

        print('-----Start Hyperparameter Tuning-----\n')
        print('  Number of Trials in pipeline: {}\n'.format(self._get_number_trials()))
        print('  Start optimization...\n')

        # Set config
        self.config_optimized = copy.deepcopy(config)
        self.config = copy.deepcopy(config)

        # Studies
        self._study_internal_save = []
        self._best_params_internal_save = []

        # Link logger internally for easier data processing
        self._link_logger()
        # Run hyperparameter optimization
        for (i, opt_conf) in enumerate(self.optimization_config):
            print(' ---Trial {} of {}---'.format(i+1, len(self.optimization_config)))
            print('   Trial configuration:\n')
            print('\n'.join(['   {} : {}'.format(k, v) for k, v in opt_conf.items()]))

            study = optuna.create_study(direction='maximize' if opt_conf['maximize'] else 'minimize', 
                                             sampler = getattr(optuna.samplers, opt_conf['sampler'])(),
                                                pruner = getattr(optuna.pruners, opt_conf['pruner'])())
            study.optimize(self.objective, n_trials=opt_conf['n_trials'])
            self._study_internal_save.append(study)
            # Save the best parameters to the optimized config
            best_params = {}
            for key, value in study.best_params.items():
                self.config_optimized[key] = value
                best_params[key] = value
            self._best_params_internal_save.append(best_params)
        
            self.logger.log_parameter_tuning(self._study_internal_save)

        print('-----End Hyperparameter Tuning-----\n')
    
    def _link_logger(self, logger: Logger = None):
        if logger is not None and self.logger is None:
            print('Logging enabled for Hyperparameter tuning')
            self.logger = logger
        if self.logger is None:
            print('Logging enabled for Hyperparameter tuning')
            self.logger = Logger(
                model_config=self.config
            )
            
    def add_trial(self, trial_config: dict):
        """
            Add a new trial to the hyperparameter optimization study.

            Arguments:
                trial_config (dict): A dictionary containing the configuration of the trial.
                    Format:
                            {  'parameter':  [
                                                {
                                                    'param': name of the parameter in the config file,
                                                    'type': parameter type in [categorical, dsicrete_uniform, float, int, loguniform, uniform],
                                                    'range': range [low, high] if continuous, discrete set [item_1,..., item_n] else
                                                }, ...
                                            ],
                                'n_trials' (int): Number of trials for the optimization of the parameters.
                                'eval_metric' (str): The evaluation metric to optimize.
                                'sampler' (str): The sampler to use. Possibble Sampler: ["TPESampler", "RandomSampler", "CmaEsSampler", "PartialFixedSampler", "NSGAIISampler", "MOTPESampler", "QMCSampler", "BruteForceSampler"]
                                'pruner; (str): The pruner to use. Possibble Pruner: ["MedianPruner, NopPruner, PatientPruner,, PercentilePruner, SuccessiveHalvingPruner, HyperbandPruner, ThresholdPruner]
                                'maximize' (bool): Whether to maximize or minimize the evaluation metric.}
                            
        """
        self.optimization_config += trial_config

    def objective(self, trial):
        """
            Objective function for hyperparameter optimization.

            Arguments:
                trial (optuna.trial.FrozenTrial): The trial to optimize.

            Returns:
                float: The objective value of the trial.
        """
        # Set trial from trial config
        set_random_seed(self.config_optimized['random_seed'])
        params = {}
        for opt_param in self.optimization_config[0]['parameter']:
            if opt_param['type'] == 'categorical':
                params[opt_param['param']] = trial.suggest_categorical(opt_param['param'], opt_param['range'])
            if opt_param['type'] == 'dsicrete_uniform':
                params[opt_param['param']] = trial.suggest_uniform(opt_param['param'], opt_param['range'][0], opt_param['range'][1])
            if opt_param['type'] == 'float':
                params[opt_param['param']] = trial.suggest_float(opt_param['param'], opt_param['range'][0], opt_param['range'][1])
            if opt_param['type'] == 'int':
                params[opt_param['param']] = trial.suggest_int(opt_param['param'], opt_param['range'][0], opt_param['range'][1])
            if opt_param['type'] == 'loguniform':
                params[opt_param['param']] = trial.suggest_loguniform(opt_param['param'], opt_param['range'][0], opt_param['range'][1])
            if opt_param['type'] == 'uniform':
                params[opt_param['param']] = trial.suggest_uniform(opt_param['param'], opt_param['range'][0], opt_param['range'][1])
        
        # Initialize model and define optimizer and loss function from config
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model, train_loader, eval_loader = self._init_trial(params)
        
        model = model.to(device)
        optimizer = initialize_optimizer(model, self.config_optimized['optimizer'])
        criterion = initialize_loss(self.config_optimized['loss'])

        
        # Force specific evaluation metric for optimization
        self.config_optimized['evaluation']['metrics'] = [self.optimization_config[0]['eval_metric']]

        # Train the model.
        score = self._train_evaluate_model(
            model = model,
            train_loader = train_loader,
            test_loader = eval_loader,
            optimizer = optimizer,
            criterion = criterion,
            trial = trial
        )

        return score
    
    def get_optimzed_config(self):
        return self.config_optimized
    
    def get_trials_data(self):
        return [s.trials_dataframe(attrs=('number', 'value', 'params')) for s in self._study_internal_save]

    ### Internal Functions ###

    def _init_trial(self, params: dict):

        def set_param(config, key, value):
            try:
                if len(key) == 1:
                    config[key[0]] = value
                else:
                    return set_param(config[key[0]], key[1:], value)
            except:
                return -1

        # Adjust the configuration such that the hyperparameters defined in params overwrite the default values.
        for k, v in params.items():
            k_split = P(k).parts
            ret = set_param(self.config_optimized, k_split, v)
            if ret == -1:
                print('Caution: Parameter {k} is not in config. Tuning will have no effect.')

        model = self._build_model()
        model = model.to(self.device)
        train_loader, test_loader = self._load_dataset()

        return model, train_loader, test_loader
    
        
    def _build_model(self):
        if self.config_optimized['model'] == 'mlp':
            model = load_mlp_model(self.config_optimized)
        return model

    def _load_dataset(self):

        if self.train_dataset is None or self.test_dataset is None:
            if self.config_optimized['dataset']['name'] == 'svhn':
                self.train_dataset, self.test_dataset = load_svhn_dataset(train_set=False)
            if self.config_optimized['dataset']['name'] == 'mnist':
                self.train_dataset, self.test_dataset = load_mnist_dataset()
        if self.config_optimized['dataset']['train_size'] < len(self.train_dataset):
            self.train_dataset, _ = data_utils.random_split(self.train_dataset, [self.config_optimized['dataset']['train_size'], len(self.train_dataset) - self.config_optimized['dataset']['train_size']])
        
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.config_optimized['batch_size'], shuffle=self.config_optimized['dataset']['train_shuffle'], drop_last=self.config_optimized['dataset']['drop_last'])
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.config_optimized['eval_batch_size'], shuffle=self.config_optimized['dataset']['eval_shuffle'], drop_last=self.config_optimized['dataset']['drop_last'])

        self.config_optimized['dataset']['val_size'] = len(self.train_dataset)
        self.config_optimized['dataset']['test_size'] = len(self.test_dataset)

        return train_loader, test_loader
    
    def _train_evaluate_model(self, model: nn.Module, 
                                train_loader: torch.utils.data.DataLoader,
                                test_loader: torch.utils.data.DataLoader,
                                optimizer: torch.optim.Optimizer, 
                                trial: optuna.trial.FrozenTrial, 
                                criterion: nn.Module, ):
        
        def __run_epoch():
            dataset_iterator = tqdm(enumerate(train_loader), total=len(train_loader))
            for i, (imgs, labels) in dataset_iterator:
                # Prepare inputs
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                imgs, labels = apply_data_preprocessing(imgs, labels, self.config_optimized)
                # Produce output
                output = model(imgs)
                # Compute loss and backpropagate
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                # Finally update all weights
                optimizer.step()
                dataset_iterator.set_description(f'Loss: {loss.item():.4f}')

        EPOCHS = self.config_optimized["num_epochs"]
        ###----Training----###
        # Set device
        # Train the model
        eval_score_epoch = []
        # Iterate for 'num_epochs' epochs
        for epoch in range(EPOCHS):
            losses = data = eval_metrics = None
            __run_epoch()
            eval_score_epoch.append(list(run_evaluation(model=model, dataset=test_loader, config=self.config_optimized).values())[0][0])
            # Pruning
            trial.report(eval_score_epoch[-1], step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return sum(eval_score_epoch) / EPOCHS
            

    def _get_number_trials(self):
        return len(self.optimization_config)

        