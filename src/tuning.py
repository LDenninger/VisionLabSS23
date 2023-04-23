import torch

import optuna

import os
import copy

from models import *
from utils import *
from .execution import *
from .evaluation import *
from .utils import *
from experiments import Logger

class HyperOpt():

    def __init__(self, optimization_config: dict = None, log: bool = False):
        
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
                                'pruner; (str): The pruner to use. Possibble Pruner: ["MedianPruner
                                'maximize' (bool): Whether to maximize or minimize the evaluation metric.
                            ]
        """
        self.log = log

        self.optimization_config = [optimization_config] if optimization_config is not None else []

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
                                            sampler = optuna.samplers.globals()[opt_conf['sampler']](),
                                                pruner = optuna.pruners.globals()[opt_conf['pruner']]())
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
    
    def _link_logger(self):
        print('Logging enabled for Hyperparameter tuning')

        self.logger = Logger(
            configuration=self.config
        )
            
    def add_trial(self, trial_config: dict):
        """
            Add a new trial to the hyperparameter optimization study.

            Arguments:
                trial_config (dict): A dictionary containing the configuration of the trial.
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
                                'maximize' (bool): Whether to maximize or minimize the evaluation metric.
                            ]
        """
        self.trial_config += trial_config

    def objective(self, trial: optuna.trial.FrozenTrial) -> float:
        """
            Objective function for hyperparameter optimization.

            Arguments:
                trial (optuna.trial.FrozenTrial): The trial to optimize.

            Returns:
                float: The objective value of the trial.
        """
        
        # Set trial from trial config
        params = {}
        for opt_param in self.trial_config[0]:
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
        self.config_optimized['evaluation']['metrics'] = self.eval_metric

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

        # Adjust the configuration such that the hyperparameters defined in params overwrite the default values.
        for k, v in params.items():
            if k in self.config:
                self.config_optimized[k] = v
            else:
                print('Caution: Parameter {k} is not in config. Tuning will have no effect.')

        model = self._build_model()
        train_loader, test_loader = self._load_dataset()

        return model, train_loader, test_loader
        
    def _build_model(self):
        if self.config['model'] == 'mlp':
            self.model = load_mlp_model(self.config)

    def _load_dataset(self):
        if self.config['dataset']['name'] == 'svhn':
            train_dataset, test_dataset = load_svhn_dataset(train_set=False)
        if self.config['dataset']['name'] == 'mnist':
            train_dataset, test_dataset = load_mnist_dataset()
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=self.config['dataset']['train_shuffle'], drop_last=self.config['dataset']['drop_last'])
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['eval_batch_size'], shuffle=self.config['dataset']['eval_shuffle'], drop_last=self.config['dataset']['drop_last'])

        self.config['dataset']['val_size'] = len(train_dataset)
        self.config['dataset']['test_size'] = len(test_dataset)

        return train_loader, test_loader
    
    def _train_evaluate_model(self, model: nn.Module, 
                                train_loader: torch.utils.data.DataLoader,
                                test_loader: torch.utils.data.DataLoader,
                                optimizer: torch.optim.Optimizer, 
                                trial: optuna.trial.FrozenTrial, 
                                criterion: nn.Module, ):
        
        def __run_epoch():
            dataset_iterator = enumerate(train_loader)
            for i, (imgs, labels) in dataset_iterator:
                # Prepare inputs
                imgs = imgs.to(device)
                labels = labels.to(device)
                imgs, labels = apply_data_preprocessing(imgs, labels, self.config_optimized)
                # Produce output
                output = model(imgs)
                # Compute loss and backpropagate
                loss = criterion(output, labels)
                optimizer.zero_grad()
                loss.backward()
                # Finally update all weights
                optimizer.step()
             
        EPOCHS = self.config_optimized["num_epochs"]
        ###----Training----###
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Train the model
        eval_score_epoch = []
        # Iterate for 'num_epochs' epochs
        for epoch in range(EPOCHS):
            losses = data = eval_metrics = None
            __run_epoch()
            eval_score_epoch.append(run_evaluation(model=model, dataset=test_loader, config=self.config_optimized))
            # Pruning
            trial.report(eval_score_epoch[-1], step=epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return sum(eval_score_epoch) / EPOCHS
            

    def _get_number_trials(self):
        return len(self.trial_config)

        