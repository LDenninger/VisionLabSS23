import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import copy
import optuna
import json
import utils
from pathlib import Path as P
import torchgadgets as tg
from tqdm import tqdm
from prettytable import PrettyTable
from models import *
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
from torchmetrics.image.fid import FrechetInceptionDistance


###--- Datasets ---###

##-- Food101 Dataset --##
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
    
class SVHNDataset(torch.utils.data.Dataset):
    """
        A simple wrapper class for PyTorch datasets to add some further functionalities.
    
    """
    def __init__(self,  transforms: list = None, train_set: bool = True):
        self.data_augmentor = None
        self.train_set = train_set
        self.dataset = tv.datasets.SVHN(root='./data/svhn', split='train' if self.train_set else 'test', download=True, transform=tv.transforms.ToTensor())
        if transforms is not None:
            self.data_augmentor = tg.data.ImageDataAugmentor(transforms)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        if self.data_augmentor is not None:
            image, label = self.data_augmentor((image, label), self.train_set)
        return image, label
    
    def __len__(self):
        return len(self.dataset)

###--- Loss Functions ---###

class GAN_discriminator_loss(torch.nn.Module):
    def __init__(self,  device=torch.device('cpu')):
        super(GAN_discriminator_loss, self).__init__()
        self.device = device

    def forward(self, x, real=True):
        target = torch.ones(x.shape[0]).float().to(self.device) if real else torch.zeros(x.shape[0]).float().to(self.device)
        return F.binary_cross_entropy(x, target)

class GAE_generator_loss(torch.nn.Module):
    def __init__(self, device=torch.device('cpu')):
        super(GAE_generator_loss, self).__init__()
        self.device = device

    def forward(self, x):
        target = torch.ones(x.shape[0]).to(self.device)
        return F.binary_cross_entropy(x, target)


###--- Run Information ---###
# These list of runs can be used to run multiple trainings sequentially.

#EXPERIMENT_NAMES = ['convnext_large']*3
#RUN_NAMES = ['norm_class', 'large_class', 'large_bn_class']

EXPERIMENT_NAMES = []
RUN_NAMES = []
EVALUATION_METRICS = ['accuracy', 'accuracy_top3', 'accuracy_top5', 'confusion_matrix', 'f1', 'recall', 'precision']
EPOCHS = [20]



class Trainer:

    ##-- Initialization --##
    def __init__(self, exp_name: str, run_name: str, suppress_output: bool = False):
        self.exp_name = exp_name
        self.run_name = run_name
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.experiment_dir = P(os.getcwd()) / 'experiments' / self.exp_name
        self.run_dir = self.experiment_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.run_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.suppress_output = suppress_output
        self.initialize_training()
       

    def initialize_training(self):
        # Config for augmentation whe nthe dataset is initially loaded, in our case only random cropping
        load_augm_config_train = utils.load_config('augm_train_preLoad') 
        load_augm_config_test = utils.load_config('augm_test_preLoad')

        # Load config for the model
        self.config = utils.load_config_from_run(self.exp_name, self.run_name)
        self.config['num_iterations'] = self.config['dataset']['train_size'] // self.config['batch_size']
        self.config['num_eval_iterations'] = self.config['dataset']['test_size'] // self.config['batch_size']

        self.conditional = self.config['model']['conditional']

        tg.tools.set_random_seed(self.config['random_seed'])
        ##-- Load Dataset --##
        # Simply load the dataset using TorchGadgets and define our dataset to apply the initial augmentations
        if self.config['dataset']['name'] == 'food101':
            train_dataset = Food101Dataset()
            test_dataset = Food101Dataset(train_set=False)
        elif self.config['dataset']['name'] == 'svhn':
            train_dataset = SVHNDataset()
            test_dataset = SVHNDataset(train_set=False)

        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True, num_workers=2)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True, num_workers=2)
        ##-- Logging --##
        # Directory of the run that we write our logs to

        # Explicitely define logger to enable TensorBoard logging and setting the log directory
        self.logger = tg.logging.Logger(log_dir=self.log_dir, checkpoint_dir=self.checkpoint_dir, model_config=self.config, save_internal=True)

        self.gan = GAN(self.config['model']).to(self.device)

        self.criterion_disc = GAN_discriminator_loss(device=self.device)
        self.criterion_gen = GAE_generator_loss(device=self.device)

        self.optim_disc = torch.optim.Adam(self.gan.discriminator.parameters(), lr=self.config['lr_discriminator'], betas=(0.5, 0.9))
        self.optim_gen = torch.optim.Adam(self.gan.generator.parameters(), lr=self.config['lr_generator'], betas=(0.5, 0.9))

        self.config['learning_rate'] = self.config['lr_discriminator']
        self.schedule_disc = tg.training.SchedulerManager(self.optim_disc, self.config)
        self.config['learning_rate'] = self.config['lr_generator']
        self.schedule_gen = tg.training.SchedulerManager(self.optim_gen, self.config)

        self.data_augmentor = tg.data.ImageDataAugmentor(config=self.config['pre_processing'])

        if self.config['model']['sample_latent'] == 'uniform':
            self.fixed_latent = torch.randn(self.config['batch_size'], self.config['model']['latent_dim'])
        elif self.config['model']['sample_latent'] == 'normal':
            self.fixed_latent = torch.normal(0.0,1.0, size=(self.config['batch_size'], self.config['model']['latent_dim']))
        if self.config['model']['conditional']:
            self.fixed_labels = F.one_hot((torch.randint(0, self.config['dataset']['num_labels'], size=(self.config['batch_size'],1)).squeeze()), num_classes=self.config['dataset']['num_labels']).float()
        self.fid = FrechetInceptionDistance(feature=64, dim=2048).to(self.device)

        
    
    ##-- Calls --##
    def reset(self):
        self.initialize_training()
    def train(self):
        self.run_training()

    ##-- Training Functions --##

    def run_training(self):
        EPOCHS = self.config['num_epochs']

        ##-- Initial Evaluation --##
        print("Initial Evaluation:")
        disc_loss, disc_loss_real, disc_loss_fake, gen_loss = self.run_epoch(0, train=False)
        self._print_result(disc_loss, disc_loss_real, disc_loss_fake, gen_loss)

        for epoch in range(EPOCHS):
            print('\nEpoch {}/{}'.format(epoch + 1, EPOCHS))

            disc_loss, disc_loss_real, disc_loss_fake, gen_loss = self.run_epoch(epoch, train=True)
            self._print_result(disc_loss, disc_loss_real, disc_loss_fake, gen_loss)

            ###--- Evaluation Epoch ---###
            if epoch % self.config['evaluation']['frequency'] == 0:
                print('\nEvaluation {}/{}'.format(epoch + 1, EPOCHS))
                disc_loss, disc_loss_real, disc_loss_fake, gen_loss = self.run_epoch(epoch, train=False)
                self._print_result(disc_loss, disc_loss_real, disc_loss_fake, gen_loss)
                

        
    def run_epoch(self,epoch, train=True):
        dataset = self.train_loader if train else self.test_loader
        iterations = self.config['num_iterations'] if train else self.config['num_eval_iterations']

        disc_loss_list = []
        disc_loss_real_list = []
        disc_loss_fake_list = []
        gen_loss_list = []
        fid_score_list = []

        if train:
            self.gan.train()
        else:
            self.gan.eval()
        ###--- Training Epoch ---###
        if not self.suppress_output:
            progress_bar = tqdm(enumerate(dataset), total=iterations)
        else:
            progress_bar = enumerate(dataset)
        for i, (img, label) in progress_bar:
            if i==iterations:
                break
            img, label = img.to(self.device), label.to(self.device)
            img_augm, label_augm = self.data_augmentor((img, label), train=train)

            ##-- Latent Space Sampling --##
            B = img.shape[0]
            if self.config['model']['sample_latent'] == 'uniform':
                latent = torch.randn(B, self.config['model']['latent_dim']).to(self.device)
            elif self.config['model']['sample_latent'] == 'normal':
                latent = torch.normal(0.0,1.0, size=(B, self.config['model']['latent_dim'])).to(self.device)
            if self.conditional:
                #samp_labels = F.one_hot((torch.randint(0, self.config['dataset']['num_labels'], size=(B,1)).squeeze()), num_classes=self.config['dataset']['num_labels']).float().to(self.device)
                disc_real_input = [img_augm, label_augm.float()]
                gen_input = [latent, label_augm.float()]
            else:
                disc_real_input = [img_augm]
                gen_input = [latent]

            ###=== Training Discriminator ===###
            if train:
                self.optim_disc.zero_grad()
            ##-- Predict real images --##
            prediction_real = self.gan.discriminate(*disc_real_input)
            ##-- Predict fake images --##
            fake_samples = self.gan.generate(*gen_input)
            #fake_samples_augm, label_augm = self.data_augmentor((fake_samples, label), train=train)
            assert fake_samples.shape == img.shape

            prediction_fake_d = self.gan.discriminate(*[fake_samples.detach(), label_augm.float().detach()] if self.conditional else [fake_samples.detach()])

            ##-- Compute discriminator loss --##
            d_loss_real = self.criterion_disc(prediction_real.view(B))
            d_loss_fake = self.criterion_disc(prediction_fake_d.view(B), real=False)
            if train:
                (d_loss_real + d_loss_fake).backward()

                ##-- Discriminator Optimization Step --##
                torch.nn.utils.clip_grad_norm_(self.gan.discriminator.parameters(), 3.0)
                self.optim_disc.step()
                
                ###=== Training Generator ===###
                self.optim_gen.zero_grad()

            ##-- Predict fake images --##
            prediction_fake_g = self.gan.discriminate(*[fake_samples, label_augm.float()] if self.conditional else [fake_samples])
            ##-- Generator Loss Computation --##
            g_loss = self.criterion_gen(prediction_fake_g.view(B))
            if train:
                g_loss.backward()

                ##-- Generator Optimization Step --##
                self.optim_gen.step()

                ###=== Finalization Training Iteration ===###

                ##-- Learning Rate Scheduler Step --##
                self.schedule_disc.step(i+1)
                self.schedule_gen.step(i+1)

                ##-- Logging --##
                log_dir = {
                    'train/disc_loss': (d_loss_real + d_loss_fake).item(),
                    'train/disc_loss_real': d_loss_real.item(),
                    'train/disc_loss_fake': d_loss_fake.item(),
                    'train/gen_loss': g_loss.item()
                }
                self.logger.log_data(data=log_dir, epoch=epoch+1, iteration=i+1, model = self.gan)
            
            if not train:
                self.fid.update(torch.round(255*img).to(dtype=torch.uint8), real=True)
                self.fid.update(torch.round(255*(fake_samples*0.5+0.5)).to(dtype=torch.uint8), real=False)
                fid_score_list.append(self.fid.compute().cpu().item())

            disc_loss_fake_list.append(d_loss_fake.item())
            disc_loss_real_list.append(d_loss_real.item())
            gen_loss_list.append(g_loss.item())
            disc_loss_list.append((d_loss_real + d_loss_fake).item())

            ##-- Progress Bar Description --##
            progress_bar.set_description("Loss: D:{:.4f}, G:{:.4f}".format((d_loss_real + d_loss_fake).item(), g_loss.item()))

            
        disc_loss = np.mean(disc_loss_list)
        disc_loss_real = np.mean(disc_loss_real_list)
        disc_loss_fake = np.mean(disc_loss_fake_list)
        gen_loss = np.mean(gen_loss_list)

        if not train:
            fid_score = np.mean(fid_score_list)
            self.logger.log_data(epoch=epoch+1, data={'test/disc_loss': disc_loss})
            self.logger.log_data(epoch=epoch+1, data={'test/disc_loss_real': disc_loss_real})
            self.logger.log_data(epoch=epoch+1, data={'test/disc_loss_fake': disc_loss_fake})
            self.logger.log_data(epoch=epoch+1, data={'test/gen_loss': gen_loss})
            self.logger.log_data(epoch=epoch+1, data={'test/fid_score': fid_score})


        return disc_loss, disc_loss_real, disc_loss_fake, gen_loss
    ##-- Evaluation Functions --##
    def generate_samples(self, class_label: int=None, batch_size=64):

        if self.config['model']['sample_latent'] == 'uniform':
            latent = torch.randn(batch_size, self.config['model']['latent_dim']).to(self.device)
        elif self.config['model']['sample_latent'] == 'normal':
            latent = torch.normal(0.0,1.0, size=(batch_size, self.config['model']['latent_dim'])).to(self.device)
        if self.conditional:
            if class_label is None:
                class_labels = torch.randint(0, self.config['dataset']['num_labels'], size=(batch_size, 1)).squeeze()
                samp_labels = F.one_hot(class_labels, num_classes=self.config['dataset']['num_labels']).float().to(self.device)
            else:
                samp_labels = F.one_hot(torch.repeat_interleave(class_label, batch_size, dim=0), num_classes=self.config['dataset']['num_labels']).float().to(self.device)
            gen_input = [latent, samp_labels]
        else:
            class_labels = torch.zeros(batch_size)
            gen_input = [latent]
        
        gen_out = self.gan.generate(*gen_input)

        # Assume normalization in the generation process
        gen_out = gen_out * 0.5 + 0.5

        gen_out = gen_out.detach().cpu()
        return gen_out, class_labels
    
    @torch.no_grad()
    def generate(self, latent, labels=None):

        self.gan.eval()
        gen_out = self.gan.generate(*[latent, labels] if self.conditional else [latent])
        gen_out = gen_out * 0.5 + 0.5
        return gen_out            

    ##-- Util Functions --##
    def _print_result(self, disc_loss, disc_loss_real, disc_loss_fake, gen_loss):
        print('Discriminator Loss: {:.4f} (real: {:.4f} fake: {:.4f}), Generator Loss: {:.4f}'.format(disc_loss, disc_loss_real, disc_loss_fake, gen_loss))

    def _sample_latent(self, size, mode: str = 'uniform'):
        if mode == 'uniform':
            latent = torch.randn(size, self.config['model']['latent_dim'])
        elif mode == 'normal':
            latent = torch.normal(0.0,1.0, size=(size, self.config['model']['latent_dim']))
        
        return latent


###--- Training ---###
# This is the function used for training all our experiments.
# The experiments are structured into the "/experiments" directory, where all TensorBoard and PyTorch files can be found

def training(exp_names, run_names):
    assert len(exp_names) == len(run_names)
    for exp_name, run_name in zip(exp_names, run_names):
        print(f' Start training...\n exp. name: \t{exp_name}, run name: \t{run_name}')
        trainer = Trainer(exp_name, run_name)
        trainer.train()



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
    
    if args.opt_study:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        optimization_study(exp_name, run_name, study_name='opt_study', n_trials=args.n)
    
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

    if args.tuning:
        assert (args.exp is not None or 'CURRENT_EXP' in os.environ) and (args.run is not None or 'CURRENT_RUN' in os.environ), 'Please provide an experiment and run name'
        exp_name = args.exp if args.exp is not None else os.environ.get('CURRENT_EXP')
        run_name = args.run if args.run is not None else os.environ.get('CURRENT_RUN')
        hyperparameter_tuning(exp_name, run_name)

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
    