import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
import copy
import optuna
import utils
import torchgadgets as tg
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import seaborn as sn
from models import ConvVAE, get_resnet_vae
from pathlib import Path as P
import json
from tqdm import tqdm
import numpy as np
import lpips
from torchmetrics.image.fid import FrechetInceptionDistance

###--- Run Information ---###
# These list of runs can be used to run multiple trainings sequentially.

EXPERIMENT_NAMES = []
RUN_NAMES = []
EVALUATION_METRICS = ['accuracy', 'accuracy_top3', 'accuracy_top5', 'confusion_matrix', 'f1', 'recall', 'precision']
EPOCHS = []

###--- Food101 Dataset ---###
# Prior to training we extracted the dataset, resized the images
# and saved them to PyTorch tensor files.
# In our experiments, data loading posed a bottleneck which made this step necessary.
# The script to extract the data can be found in the file 'prepare_data.py'
class Food101Dataset(torch.utils.data.Dataset):
    """
        A simple wrapper class for PyTorch datasets to add some further functionalities.
    
    """
    def __init__(self,  transforms: list = None, train_set: bool = True):
        self.data_augmentor = None
        self.train_set = train_set
        if train_set:
            self.data_dir = P(os.getcwd()) / 'data' / 'food101_processed' / 'train'
        else:
            self.data_dir = P(os.getcwd()) / 'data' / 'food101_processed' / 'test'

        with open(str(self.data_dir / 'labels.json'), 'r') as f:
            self.labels = json.load(f)

        if transforms is not None:
            self.data_augmentor = tg.data.ImageDataAugmentor(transforms)

    def __getitem__(self, index):
        image = torch.load(str(self.data_dir / f'img_{str(index).zfill(6)}.pt'))
        label = self.labels[index]
        if self.data_augmentor is not None:
            image, label = self.data_augmentor((image, label), self.train_set)
        return image, label
    
    def __len__(self):
        return len(self.labels)
    
###--- Loss Functions ---###

##-- Structural Similarity Index Measure --##

# The SSIM is a bit more elaborate loss functions which measures the similarity of images
# based on luminance, contrast and structure.
# The output  is in the range of [-1,1], where 1 indicates highly similar images.
# We normalize it to [0,1] and invert it such that 0 indicates similar images.
class SSIM_KLD_Loss(nn.Module):

    def __init__(self, lambda_kld=1e-3, device='cpu'):
        super(SSIM_KLD_Loss, self).__init__()
        self.lambda_kld = lambda_kld
        self.ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.device = device

    def forward(self, output, target, mu, log_var):
        # SSIM output [-1,1] --> [0,1]
        recons_loss = 1-(1+self.ssim(output, target))/2
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)
        loss = recons_loss + self.lambda_kld * kld

        return loss, (recons_loss, kld)
    
##-- MSE Sum Loss --##
# MSE reconstruction loss with KL divergence regularization term.
# Here we take the sum within an image and average over a batch.
    
class MSE_SUM_KLD_Loss(nn.Module):


    def __init__(self, lambda_kld=1e-3, device='cpu'):
        super(MSE_SUM_KLD_Loss, self).__init__()
        self.lambda_kld = lambda_kld
        self.device = device

    def forward(self, output, target, mu, log_var):
        recons_loss = F.mse_loss(
                output.reshape(target.shape[0], -1),
                target.reshape(target.shape[0], -1),
                reduction="none",
            ).sum(dim=-1).mean(dim=0)
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)
        loss = recons_loss + self.lambda_kld * kld

        return loss, (recons_loss, kld)

##-- MSE Mean Loss --##
# MSE reconstruction loss with KL divergence regularization term.
# Here we take the average over all errors.

class MSE_MEAN_KLD_Loss(nn.Module):

    def __init__(self, lambda_kld=1e-3, device='cpu'):
        super(MSE_MEAN_KLD_Loss, self).__init__()
        self.lambda_kld = lambda_kld
        self.device = device

    def forward(self, output, target, mu, log_var):
        recons_loss = F.mse_loss(
                output.reshape(target.shape[0], -1),
                target.reshape(target.shape[0], -1),
            )
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)
        loss = recons_loss + self.lambda_kld * kld

        return loss, (recons_loss, kld)
    
##-- Beta Loss --##
# Loss functions used for beta-VAEs.
# It is higly similar to the previous loss except that we have an additional beta term.
# Beta should we be set > 1 to encourage a disentangled representation in the latent space
class BetaLoss(nn.Module):


    def __init__(self, lambda_kld=1e-3, beta=4):
        super(BetaLoss, self).__init__()
        self.lambda_kld = lambda_kld
        self.beta = beta

    def forward(self, output, target, mu, log_var):
        recons_loss = F.mse_loss(
                output.reshape(target.shape[0], -1),
                target.reshape(target.shape[0], -1),
            )
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)
        loss = recons_loss + self.lambda_kld * self.beta * kld

        return loss, (recons_loss, kld)
    
##-- MSE-LPIPS Loss --##
# Similar to the MSE sum loss with an additional regularization term that evaluates the similarity of the images
class MSE_MEAN_LPIPS_Loss(nn.Module):

    def __init__(self, lambda_kld=1e-3, lambda_lpips=1e-2, device='cpu'):
        super(MSE_MEAN_LPIPS_Loss, self).__init__()
        self.lambda_kld = lambda_kld
        self.lambda_lpips = lambda_lpips
        self.lpips_loss = lpips.LPIPS(net='vgg').cuda()
        self.device = device

    def forward(self, output, target, mu, log_var):
        recons_loss = F.mse_loss(
                output.reshape(target.shape[0], -1),
                target.reshape(target.shape[0], -1),
            )
        kld = (-0.5 * (1 + log_var - mu**2 - log_var.exp()).sum(dim=1)).mean(dim=0)
        output = (output-1)*2
        target = (target-1)*2
        lpips_score = torch.mean(self.lpips_loss(output, target))

        loss = recons_loss + self.lambda_kld * kld + self.lambda_lpips * lpips_score
        return loss, (recons_loss, kld)

###--- Training ---###
# This is the function used for training all our experiments.
# The experiments are structured into the "/experiments" directory, where all TensorBoard and PyTorch files can be found

def training(exp_names, run_names):
    assert len(exp_names) == len(run_names)
    for exp_name, run_name in zip(exp_names, run_names):
        ##-- Load Config --##
        # Load the config from the run directory
        # All interactions with the experiments directory should be performed via the utils package

        # Config for augmentation whe nthe dataset is initially loaded, in our case only random cropping
        
        load_augm_config_train = utils.load_config('augm_train_preLoad') 
        #load_augm_config_test = utils.load_config('augm_test_preLoad')

        # Load config for the model
        config = utils.load_config_from_run(exp_name, run_name)
        config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']
        config['num_eval_iterations'] = config['dataset']['test_size'] // config['batch_size']

        #config['num_iterations'] = 400
        tg.tools.set_random_seed(config['random_seed'])
        ##-- Load Dataset --##
        # Simply load the dataset using TorchGadgets and define our dataset to apply the initial augmentations
        train_dataset = Food101Dataset(transforms=load_augm_config_train)
        test_dataset = Food101Dataset(train_set=False)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4)
        ##-- Logging --##
        # Directory of the run that we write our logs to
        log_dir = os.path.join(os.getcwd(),'experiments', exp_name, run_name, 'logs')
        checkpoint_dir = os.path.join(os.getcwd(), 'experiments', exp_name, run_name, 'checkpoints')

        # Explicitely define logger to enable TensorBoard logging and setting the log directory
        logger = tg.logging.Logger(log_dir=log_dir, checkpoint_dir=checkpoint_dir, model_config=config, save_internal=True, save_external=True)
        if config['model']['type'] in ['vae', 'cond_vae']:
            model = ConvVAE(config['model'])
        if config['model']['type'] in 'resnet_vae':
            model = get_resnet_vae(config['model'])

        if config['loss']['type'] == 'ssim':
            criterion = SSIM_KLD_Loss(lambda_kld=config['loss']['lambda_kld'], device='cuda')
        if config['loss']['type'] == 'mse_sum':
            criterion = MSE_SUM_KLD_Loss(lambda_kld=config['loss']['lambda_kld'])
        if config['loss']['type'] == 'mse_mean':
            criterion = MSE_MEAN_KLD_Loss(lambda_kld=config['loss']['lambda_kld'])
        if config['loss']['type'] == 'beta':
            criterion = BetaLoss(lambda_kld=config['loss']['lambda_kld'], beta=config['loss']['beta'])
        if config['loss']['type'] == 'lpips':
            criterion = MSE_MEAN_LPIPS_Loss(lambda_kld=config['loss']['lambda_kld'], lambda_lpips=config['loss']['lambda_lpips'])

        optimizer = tg.training.initialize_optimizer(model, config)
        scheduler = tg.training.SchedulerManager(optimizer, config)

        data_augmentor = tg.data.ImageDataAugmentor(config=config['pre_processing'])

        if config['model']['type'] in ['vae', 'resnet_vae']:
            train_vae(config=config, 
                        model=model, 
                            logger=logger, 
                                criterion=criterion, 
                                    optimizer=optimizer,
                                        data_augmentor=data_augmentor,
                                            train_loader=train_loader, 
                                                val_loader=test_loader, 
                                                    scheduler=scheduler,
                                                        suppress_output=False)
        



##-- VAE Training Script --##
def train_vae(model, config, train_loader, val_loader, optimizer, criterion,  data_augmentor, scheduler=None, logger=None, suppress_output=False):

    ###--- Hyperparameters ---###

    EPOCHS = config['num_epochs']
    ITERATIONS = config['num_iterations']

    evaluation_config = config['evaluation']

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ###--- Initial Evaluation ---###
    # Evaluate the untrained model to have some kind of base line for the training progress
    print('Initial Evluation')
    evaluation_metrics, eval_loss, mse, kld = run_vae_evaluation(model,data_augmentor,val_loader,config=config, criterion=criterion, suppress_output=False)

    # Log evaluation data
    if logger is not None:
        logger.log_data(epoch=0, data=evaluation_metrics)
        logger.log_data(epoch=0, data={'eval_loss': eval_loss})
        logger.log_data(epoch=0, data={'mse': mse})
        logger.log_data(epoch=0, data={'kld': kld})

    if logger is not None and logger.save_internal:
        logs = logger.get_last_log()
        print("".join([(f' {key}: {value},') for key, value in logs.items()]))


    ###--- Training ---###
    # Train for EPOCHES epochs and evaluate the model according to the pre-defined frequency
    for epoch in range(EPOCHS):
        print('\nEpoch {}/{}'.format(epoch + 1, EPOCHS))
        model.train()

        outputs = []
        targets = []
        #training_metrics = []

        ###--- Training Epoch ---###
        if not suppress_output:
            progress_bar = tqdm(enumerate(train_loader), total=config['num_iterations'])
        else:
            progress_bar = enumerate(train_loader)
        for i, (img, label) in progress_bar:
            if i==config['num_iterations']:
                    break
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            # Apply data augmentation and pre-processing
            img_augm, label_augm = data_augmentor((img, label))
            # Zero gradients
            optimizer.zero_grad()
            # Compute output of the model
            if config['model']['conditional']:
                output, (z, mu, sigma) = model(img_augm, label_augm)
            else:
                output, (z, mu, sigma) = model(img_augm)

            # Compute loss
            loss, (mse, kld) = criterion(output.float(), img.float(), mu, sigma)
            # Backward pass to compute the gradients wrt to the loss
            loss.backward()
            # Update weights
            optimizer.step()
            # Log training data
            if logger is not None:
                logger.log_data(data={'train_loss': loss.item(), 'mse': mse.item(), 'kld': kld.item()}, epoch=epoch+1, iteration=i+1, model = model, optimizer = optimizer)
            #tr_metric = eval_resolve(output, label, config)['accuracy'][0]
            #raining_metrics.append(tr_metric)
            if not suppress_output:
                progress_bar.set_description(f'Loss (recon/kld): {loss.cpu().item():.4f} ({mse.cpu().item():.4f}/{kld.cpu().item():.4f})')
            if scheduler is not None:
                # Learning rate scheduler takes a step
                    scheduler.step(i+1)
            outputs.append(1.0)
        ###--- Evaluation Epoch ---###
        if epoch % evaluation_config['frequency'] == 0:
            evaluation_metrics, eval_loss, mse, kld = run_vae_evaluation(model,data_augmentor,val_loader,config, criterion=criterion, suppress_output=False)

        # Log evaluation data
        if logger is not None:
            #logger.log_data(epoch=epoch+1, data={'train_accuracy': training_metrics})
            logger.log_data(epoch=epoch+1, data=evaluation_metrics)
            logger.log_data(epoch=epoch+1, data={'eval_loss': eval_loss})
            logger.log_data(epoch=epoch+1, data={'eval_mse': mse})
            logger.log_data(epoch=epoch+1, data={'eval_kld': kld})
        
        # If the logger is activated and saves the data internally we can print out the data after each epoch
        if logger is not None and logger.save_internal:
            logs = logger.get_last_log()
            print(", ".join([(f'{key}: {value}') for key, value in logs.items()]))
        

    

###--- Evaluation ---###
# This is the function used for evaluation.
# The different implementations of the evaluation metrics can be found in the TorchGadgets package

@torch.no_grad()
def run_vae_evaluation( model: torch.nn.Module, 
                    data_augmentor,
                    dataset: torch.utils.data.DataLoader, 
                    config: dict,
                    evaluation_metrics = None,
                    criterion = None,
                    suppress_output: bool = False):
    """
        Runs evaluation of the given model on the given dataset.

        Arguments:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            config (dict): The configuration dictionary.
            device (str, optional): The device to evaluate on. Defaults to "cpu".

    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_iterations = config['num_eval_iterations'] if config['num_eval_iterations'] != -1 else len(dataset)
    
    # Setup model for evaluation
    model.eval()
    model.to(device)
    eval_metrics = {}
    if suppress_output:
        progress_bar = enumerate(dataset)
    else:
        progress_bar = tqdm(enumerate(dataset), total=num_iterations)
        progress_bar.set_description(f'Evaluation:')
    outputs = []
    targets = []
    losses = []
    mse_list = []
    kld_list = []
    for i, (imgs, labels) in progress_bar:
        if i==num_iterations:
            break
        imgs, labels = imgs.to(device), labels.to(device)
        # apply preprocessing surch as flattening the imgs and create a one hot encodinh of the labels
        img_augm, labels_augm = data_augmentor((imgs, labels), train=False)
        if config['model']['conditional']:
            output, (z, mu, sigma) = model(img_augm, labels_augm)
        else:
            output, (z, mu, sigma) = model(img_augm)
            # Compute loss
        outputs.append(output.cpu())
        targets.append(labels.cpu())
        if criterion is not None:
            loss, (mse, kld) = criterion(output.float(), imgs.float(), mu, sigma)
            losses.append(loss.cpu().item())
            mse_list.append(mse.cpu().item())
            kld_list.append(kld.cpu().item())

    eval_metrics = tg.evaluation.evaluate(torch.stack(outputs, dim=0), torch.stack(targets, dim=0), config=config, metrics=evaluation_metrics)

    if criterion is None:
        return eval_metrics
    
    return eval_metrics, sum(losses) / len(losses), sum(mse_list) / len(mse_list), sum(kld_list) / len(kld_list), 


@torch.no_grad()
def compute_frechet_inception_score(exp_name, run_name, data_augmentor, model=None, dataset=None, conditional=False):

    ##-- Hyperparameters for Evaluation --##
    dim = 2048
    feature = 64
    device = ('cuda' if torch.cuda.is_available() else 'cpu')


    # Load config for the model
    config = utils.load_config_from_run(exp_name, run_name)
    config['num_iterations'] = config['dataset']['train_size'] // config['batch_size']
    config['num_eval_iterations'] = config['dataset']['test_size'] // config['batch_size']

    # Load the model
    if model is None:
        if config['model']['type'] in ['vae', 'cond_vae']:
            model = ConvVAE(config['model'])
        if config['model']['type'] == 'resnet_vae':
            model = get_resnet_vae(config['model'])

    model = model.to(device)
    model.eval()
    # Load the dataset
    if dataset is None:
        test_dataset = Food101Dataset(train_set=False)
        dataset = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True, num_workers=4)

    # Initialize FID metric
    fid = FrechetInceptionDistance(feature=feature, dim=dim).to(device)

    progress_bar = tqdm(enumerate(dataset), total=len(dataset))

    score = []
    for i, (img, label) in progress_bar:
        img = img.to(device)
        label = label.to(device)

        img_augm, label_augm = data_augmentor((img, label))
        if conditional:
            label = F.one_hot(label, 101)
            pred, (y, mu, logvar) = model(img_augm, label)
        else:
            pred, (y, mu, logvar) = model(img_augm)

        fid.update(torch.round(255*img).to(dtype=torch.uint8), real=True)
        fid.update(torch.round(255*pred).to(dtype=torch.uint8), real=False)
        score.append(fid.compute().cpu().item())

    return np.mean(score)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()

    # Flags to signal which function to run
    argparser.add_argument('--train', action='store_true', default=False, help='Train the model')
    argparser.add_argument('--evaluate', action='store_true', default=False, help='Evaluate the model')
    argparser.add_argument('--tuning', action='store_true', default=False, help='Tune the hyperparameters')

    argparser.add_argument('--augm_study', action='store_true', default=False, help='Run the augmentation study')
    argparser.add_argument('--opt_study', action='store_true', default=False, help='Run the augmentation study')

    argparser.add_argument('--init_exp', action='store_true', default=False, help='Initialize a new experiment')
    argparser.add_argument('--init_run', action='store_true', default=False, help='Initialize a new run')

    argparser.add_argument('--copy_conf', action='store_true', default=False, help='Load a configuration file to run')

    argparser.add_argument('--clear_logs', action='store_true', default=False, help='Clear the logs of a given run')

    # Hyperparameters 
    argparser.add_argument('-exp', type=str, default=None, help='Experiment name')
    argparser.add_argument('-run', type=str, default=None, help='Run name')
    argparser.add_argument('-conf', type=str, default=None, help='Config name')

    # Additional parameter
    argparser.add_argument('-n', type=int, default=None)
    

    ##-- Function Calls --##
    # Here we simply determien which function to call and how to set the experiment and run name

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
    if args.clear_logs:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ) and args.conf is not None, 'Please provide an experiment and run name and the name of the config file'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        utils.clear_logs(exp_name, run_name)


    if args.train:
        assert ((args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ)) or (len(EXPERIMENT_NAMES)!=0 and len(RUN_NAMES)!=0), 'Please provide an experiment and run name'
        if len(EXPERIMENT_NAMES) != 0:
            assert len(EXPERIMENT_NAMES) == len(RUN_NAMES), 'Length of experiment and run list must be the same'
            exp_name = EXPERIMENT_NAMES
            run_name = RUN_NAMES
        else:       
            exp_name = [args.exp] if args.exp is not None else [os.environ.get('CURRENT_EXP')]
            run_name = [args.run] if args.run is not None else [os.environ.get('CURRENT_RUN')]
        training(exp_name, run_name)

    if args.evaluate:
        assert ((args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ)) or (len(EXPERIMENT_NAMES)!=0 and len(RUN_NAMES)!=0), 'Please provide an experiment and run name'
        if len(EXPERIMENT_NAMES) != 0:
            assert len(EXPERIMENT_NAMES) == len(RUN_NAMES), 'Length of experiment and run list must be the same'
            exp_name = EXPERIMENT_NAMES
            run_name = RUN_NAMES
        else:       
            exp_name = [args.exp] if args.exp is not None else [os.environ.get('CURRENT_EXP')]
            run_name = [args.run] if args.run is not None else [os.environ.get('CURRENT_RUN')]
        evaluation(exp_name, run_name)
    