import argparse
import os
from typing import Optional, Sized

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Sampler

import numpy as np

import copy
import optuna

import utils

import torchgadgets as tg

from prettytable import PrettyTable

import matplotlib.pyplot as plt

import seaborn as sn

from tqdm import tqdm

import json 

from PIL import Image

import copy

from models import SiameseModel

from pathlib import Path as P

from pytorch_metric_learning import losses as pml_losses
from pytorch_metric_learning import miners as pml_miners


###--- Run Information ---###
# These list of runs can be used to run multiple trainings sequentially.

EXPERIMENT_NAMES = []
RUN_NAMES = []
EVALUATION_METRICS = []
EPOCHS = []

###--- Datasets ---###

class Market1501(Dataset):
    '''
        A dataset wrapper class for the Market1501 dataset.
        This class was adapted from https://github.com/CoinCheung/triplet-reid-pytorch/blob/master/datasets/Market1501.py

        The dataset has to be initially downloaded using the script: download_dataset.sh.
        Unfortunately it seems that the server is down.
        The other option is to download it from a Google drive: https://drive.google.com/file/d/0B8-rUzbwVRk0c054eEozWG9COHM/view?pli=1&resourcekey=0-8nyl7K9_x37HlQm34MmrYQ

        Before using the DataLoader one needs to run the script: build_meta_file.sh.
        This script reads the data and builds a meta file that contains the image names for each person to quickly load positive samples.

        The problem of the dataset is the test set that does not contain information about the persons.
        We take the images from the query directory as the test set.
    '''
    def __init__(self, data_path="data/Market-1501", is_train=True, transforms=None):
        super(Market1501, self).__init__()
        self.data_path = os.path.join(data_path, 'bounding_box_train' if is_train else 'bounding_box_test')
        
        with open(os.path.join(data_path, 'train_meta_dir.json' if is_train else 'test_meta_dir.json'), 'r') as f:
            self.meta_dir = json.load(f)
        with open(os.path.join(data_path, 'train_img_paths.json' if is_train else 'test_img_paths.json'), 'r') as f:
            self.img_paths = json.load(f)

        self.lb_ids = [int(el.split('_')[0]) for el in self.img_paths]
        self.lb_cams = [int(el.split('_')[1][1]) for el in self.img_paths]

        self.img_paths = [os.path.join(self.data_path, img_path) for img_path in self.img_paths]


        self.transforms = transforms
 

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx=None, label=None):
        assert idx is not None or label is not None, 'Please provide either idx or label'
        if idx is not None and label is not None:
            assert idx < len(self.meta_dir[label]), "Please provide an index corresponding to the index of the image within the class"
            img_path = self.meta_dir[str(label).zfill(4)][idx]
        elif idx is not None:
            label = self.lb_ids[idx]
            img_path = self.img_paths[idx]
        elif label is not None:
            img_path = np.random.choice(self.img_paths[str(label).zfill(4)])

        img = Image.open(img_path)
        img = self.transforms(img)

        return img, label
    
class TripletDataset(Dataset):
    def __init__(self, dataset, data_path="data/Market-1501", is_train=True):
        super(TripletDataset, self).__init__()
        self.dataset = dataset
        with open(os.path.join(data_path, 'train_meta_dir.json' if is_train else 'test_meta_dir.json'), 'r') as f:
            self.meta_dir = json.load(f)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):

        anchor_img, anchor_label = self.dataset[idx]

        positive_img_idx = np.random.choice(list(self.meta_dir[str(anchor_label)].keys()))
        while positive_img_idx == idx:
            positive_img_idx = np.random.choice(list(self.meta_dir[str(anchor_label)].keys()))
        positive_img, positive_label = self.dataset[int(positive_img_idx)]

        negative_img_pid = np.random.choice(list(self.meta_dir.keys()))
        while negative_img_pid == anchor_label:
            negative_img_pid = np.random.choice(list(self.meta_dir.keys()))

        negative_img_id = np.random.choice(list(self.meta_dir[str(negative_img_pid)].keys()))
        negative_img, negative_label = self.dataset[int(negative_img_id)]

        return (anchor_img, positive_img, negative_img), (anchor_label, positive_label, negative_label)
    
class DoubleDataset(Dataset):
    def __init__(self, dataset, data_path="data/Market-1501", is_train=True):
        super(DoubleDataset, self).__init__()
        self.dataset = dataset
        with open(os.path.join(data_path, 'train_meta_dir.json' if is_train else 'test_meta_dir.json'), 'r') as f:
            self.meta_dir = json.load(f)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        anchor_img, anchor_label = self.dataset[idx]

        positive_img_idx = np.random.choice(list(self.meta_dir[str(anchor_label)].keys()))
        while positive_img_idx == idx:
            positive_img_idx = np.random.choice(list(self.meta_dir[str(anchor_label)].keys()))
        positive_img, positive_label = self.dataset[int(positive_img_idx)]

        return (anchor_img, positive_img), (anchor_label, positive_label)
    
class NPairSampler(Sampler):
    def __init__(self, N, data_source, data_path="data/Market-1501", drop_last=True, is_train=True) -> None:
        super().__init__(data_source)

        self.N = N
        self.dataset_length = data_source
        self.iterations = self.dataset_length // N if drop_last else self.dataset_length // N + 1
        with open(os.path.join(data_path, 'train_meta_dir.json' if is_train else 'test_meta_dir.json'), 'r') as f:
            self.meta_dir = json.load(f)
        self.class_size = {}
        for key, value in self.meta_dir.items():
            self.class_size[int(key)] = len(value.keys())
        self.labels = self.meta_dir.keys()
        assert  self.N <= len(self.meta_dir.keys()), "Please provide N larger than the number of classes"

    def __iter__(self):
        avail_data = copy.deepcopy(self.meta_dir)
        removed_labels = []

        indices = []
        for i in range(self.iterations):
            ext_req = False
            avail_keys = list(avail_data.keys())

            if len(avail_keys) < self.N:
                ext_req = True
                diff = self.N - len(avail_keys)
                np.random.shuffle(removed_labels)
                ext_ind = removed_labels[:diff]
                avail_keys = avail_keys + ext_ind
                np.random.shuffle(avail_keys)
                random_labels = avail_keys
            else:
                np.random.shuffle(avail_keys)
                random_labels = avail_keys[:self.N]
            for label in random_labels:
                if ext_req and label in ext_ind:
                    index = np.random.choice(list(self.meta_dir[label].keys()))
                else:
                    index = np.random.choice(list(avail_data[label].keys()))
                    avail_data[label].pop(index)

                    if len(avail_data[label]) == 0:
                        avail_data.pop(label)
                        removed_labels.append(label)

                indices.append(int(index))
                    
        return iter(indices)

    def __len__(self):
        return self.dataset_length

###--- Loss Functions ---###

class TripletLoss(nn.Module):
    """ Implementation of the triplet loss function """
    def __init__(self, margin=0.2, negative_mining=False, reduce="mean"):
        """ Module initializer """
        assert reduce in ["mean", "sum"]
        super().__init__()
        self.margin = margin
        self.reduce = reduce
        self.negative_mining = negative_mining
        return
        
    def forward(self, anchor, positive, negative=None, labels=None):
        if self.negative_mining:
            assert labels is not None, "Please provide a label for the negative samples"
            return self.compute_loss_with_negative_mining(anchor, positive, labels)
        else:
            assert negative is not None, "Please provide a negative sample"
            return self.compute_loss(anchor, positive, negative)
    

    def compute_loss(self, anchor, positive, negative):
        """ Computing pairwise distances and loss functions """
        # L2 distances
        d_ap = (anchor - positive).pow(2).sum(dim=-1)
        d_an = (anchor - negative).pow(2).sum(dim=-1)
        
        # triplet loss function
        loss = (d_ap - d_an + self.margin)
        loss = torch.maximum(loss, torch.zeros_like(loss))
        
        # averaging or summing      
        loss = torch.mean(loss) if(self.reduce == "mean") else torch.sum(loss)
      
        return loss
    
    def compute_loss_with_negative_mining(self, anchor, positive, labels):
        d_ap = (anchor - positive).pow(2).sum(dim=-1)
        d_an = torch.zeros_like(d_ap)
        anchor_pairwise_dist = torch.cdist(anchor, anchor, p=2)

        for anchor_id in range(anchor_pairwise_dist.shape[0]):
            non_same_label_index = torch.nonzero(labels != labels[anchor_id]).squeeze()
            larger_than_pos_index = torch.nonzero(anchor_pairwise_dist[anchor_id][non_same_label_index] > d_ap[anchor_id]).squeeze()
            if anchor_pairwise_dist[anchor_id][non_same_label_index][larger_than_pos_index].numel()==0:
                minimum_distance = torch.min(anchor_pairwise_dist[anchor_id][non_same_label_index]) # Allow violation
            else:
                minimum_distance = torch.min(anchor_pairwise_dist[anchor_id][non_same_label_index][larger_than_pos_index])
            d_an[anchor_id] = minimum_distance
        
        # triplet loss function
        loss = (d_ap - d_an + self.margin)
        loss = torch.maximum(loss, torch.zeros_like(loss))
        
        # averaging or summing      
        loss = torch.mean(loss) if(self.reduce == "mean") else torch.sum(loss)
      
        return loss
    
class AngularLoss(nn.Module):
    """
        Implementation of the angular loss function using the pytorch-metric-learning library.
    """

    def __init__(self, alpha=40):
        super().__init__()
        self.loss = pml_losses.AngularLoss(alpha=alpha)
        self.miner = pml_miners.AngularMiner()
        return
    
    def forward(self, anchor, positive, labels):
        input = torch.cat((anchor, positive), dim=0)
        labels = torch.cat((labels, labels), dim=0)
        miner_output = self.miner(input, labels)
        loss = self.loss(input, labels, miner_output)

        return loss

class NPairsLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = pml_losses.NPairsLoss()
    def forward(self, anchor, positive, labels):
        input = torch.cat((anchor, positive), dim=0)
        labels = torch.cat((labels, labels), dim=0)
        loss = self.loss(input, labels)
        return loss
    

###--- Training ---###
# This is the function used for training all our experiments.
# The experiments are structured into the "/experiments" directory, where all TensorBoard and PyTorch files can be found

class Trainer:
    def __init__(self, exp_name, run_name, logging=True):
        self.exp_name = exp_name
        self.run_name = run_name
        self.logging = logging
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.experiment_dir = P(os.getcwd()) / 'experiments' / self.exp_name
        self.run_dir = self.experiment_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = self.run_dir / 'logs'
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.run_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.initialize_training()


    ##-- Main Calls --##
    def fit(self):
        self.training()

    def reset(self):
        self.initialize_training()
    
    def load_checkpoint(self, epoch):
        utils.load_model_from_checkpoint(self.exp_name, self.run_name, self.model, epoch, optimizer=None)

    @torch.no_grad()
    def get_embedding_vector(self, input):
        self.model.eval()
        return self.model(input.to(self.device)).cpu()

    
    def initialize_training(self):
        # Load config for the model

        self.config = utils.load_config_from_run(self.exp_name, self.run_name)
        self.config['num_iterations'] = self.config['dataset']['train_size'] // self.config['batch_size']
        self.config['num_eval_iterations'] = self.config['dataset']['test_size'] // self.config['batch_size']

        tg.tools.set_random_seed(self.config['random_seed'])
        ##-- Load Dataset --##
        # Simply load the dataset using TorchGadgets and define our dataset to apply the initial augmentations
        base_train_dataset = Market1501(transforms=tv.transforms.ToTensor())
        base_test_dataset = Market1501(transforms=tv.transforms.ToTensor(), is_train=False)

        if self.config['loss']['type'] == 'TripletLoss' and not self.config['loss']['mining']:
            train_dataset = TripletDataset(dataset=base_train_dataset)
            test_dataset = TripletDataset(dataset=base_test_dataset, is_train=False)
            self.training_function = self.run_epoch_triplet
        else:
            self.triplet = False
            train_dataset = DoubleDataset(dataset=base_train_dataset)
            test_dataset = DoubleDataset(dataset=base_test_dataset, is_train=False)
            self.training_function = self.run_epoch_double

        num_workers = 2
        
        if self.config['loss']['type'] in ['NPairsLoss']:
            train_sampler = NPairSampler(N = self.config['batch_size'], data_source = self.config['dataset']['train_size'])
            test_sampler = NPairSampler(N = self.config['batch_size'], data_source = self.config['dataset']['test_size'])
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], drop_last=True,sampler=train_sampler, num_workers=num_workers)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], drop_last=True, sampler=test_sampler, num_workers=num_workers)


        else:
            self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True, num_workers=num_workers)
            self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True, drop_last=True, num_workers=num_workers)
        ##-- Logging --##
        # Directory of the run that we write our logs to
        self.model = SiameseModel(emb_dim=self.config['model']['emb_dim'], pretrained=self.config['model']['pretrained'])
        self.model = self.model.to(self.device)
        # Explicitely define logger to enable TensorBoard logging and setting the log directory
        if self.logging:
            self.logger = tg.logging.Logger(log_dir=self.log_dir, checkpoint_dir=self.checkpoint_dir, model_config=self.config, save_internal=True)

        if self.config['loss']['type'] == 'TripletLoss':
            self.criterion = TripletLoss(margin=self.config['loss']['margin'], reduce=self.config['loss']['reduce'], negative_mining=self.config['loss']['mining'])
        elif self.config['loss']['type'] == 'AngularLoss':
            self.criterion = AngularLoss(alpha=self.config['loss']['alpha'])
        elif self.config['loss']['type'] == 'NPairsLoss':
            self.criterion = NPairsLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'], betas=(0.5, 0.9))

        self.data_augmentor = tg.data.ImageDataAugmentor(config=self.config['pre_processing'])

    def run_epoch_triplet(self, epoch, is_train=True):
        
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        progress_bar = tqdm(enumerate(self.train_loader), total=self.config['num_iterations'])

        losses = []

        for i, ((anchor, positive, negative), (anchor_lbl, positive_lbl, negative_lbl)) in progress_bar:
            # setting inputs to GPU
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)
            negative = negative.to(self.device)

            # forward pass and triplet loss
            if self.config['model']['stack_input']:
                input = torch.cat((anchor, positive, negative), dim=0)
                label = torch.cat((anchor_lbl, positive_lbl, negative_lbl), dim=0)
                input, label = self.data_augmentor((input, label))
                embedding = self.model(input)
                anchor_emb = embedding[:self.config['batch_size']]
                positive_emb = embedding[self.config['batch_size']:2*self.config['batch_size']]
                negative_emb = embedding[2*self.config['batch_size']:]
            else:
                anchor, anchor_lbl = self.data_augmentor((anchor, anchor_lbl))
                positive, positive_lbl = self.data_augmentor((positive, positive_lbl))
                negative, negative_lbl = self.data_augmentor((negative, negative_lbl))
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)
                negative_emb = self.model(negative)

            loss = self.criterion(anchor_emb, positive_emb, negative=negative_emb)

            # backward pass
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.logging:
                    log_dir = {
                        'train/loss': loss.item(),
                    }
                    self.logger.log_data(data=log_dir, epoch=epoch+1, iteration=i+1, model = self.model)
            if i == 0:
                loss_smooth = loss.item()
            else:
                loss_smooth = 0.8*loss_smooth + 0.2*loss.item()
            losses.append(loss.item())

            progress_bar.set_description(f'Loss: {loss_smooth}')

        return losses

    def run_epoch_double(self, epoch, is_train=True):
        if is_train:
            self.model.train()
        else:
            self.model.eval()

        progress_bar = tqdm(enumerate(self.train_loader), total=self.config['num_iterations'])

        losses = []

        for i, ((anchor, positive), (anchor_lbl, positive_lbl)) in progress_bar:
            # setting inputs to GPU
            anchor = anchor.to(self.device)
            positive = positive.to(self.device)

            # forward pass
            if self.config['model']['stack_input']:
                input = torch.cat((anchor, positive), dim=0)
                embedding = self.model(input)
                anchor_emb = embedding[:self.config['batch_size']]
                positive_emb = embedding[self.config['batch_size']:]
            else:
                anchor_emb = self.model(anchor)
                positive_emb = self.model(positive)

            loss = self.criterion(anchor_emb, positive_emb, labels=anchor_lbl)

            # backward pass
            if is_train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                if self.logging:
                    log_dir = {
                        'train/loss': loss.item(),
                    }
                    self.logger.log_data(data=log_dir, epoch=epoch+1, iteration=i+1, model = self.model)

            if i == 0:
                loss_smooth = loss.item()
            else:
                loss_smooth = 0.8*loss_smooth + 0.2*loss.item()
            losses.append(loss.item())

            progress_bar.set_description(f'Loss: {loss_smooth}')

        return losses

    def training(self):
        # Initial Evaluation
        print(f'Initial Evaluation:')
        losses = self.training_function(epoch=-1, is_train=False)
        if self.logging:
            self.logger.log_data(data={'test/loss': np.mean(losses)}, epoch=0)
        print(' Eval Loss: {:.4f}'.format(np.mean(losses)))

        for epoch in range(self.config['num_epochs']):
            p_str = ''
            print(f'\nEpoch {epoch + 1}/{self.config["num_epochs"]}')
            losses = self.training_function(epoch=epoch, is_train=True)
            p_str += f' Train Loss: {np.mean(losses)}'

            # Evaluation epoch
            if epoch % self.config['evaluation']['frequency'] == 0:
                losses = self.training_function(epoch=epoch, is_train=False)
                if self.logging:
                    self.logger.log_data(data={'test/loss': np.mean(losses)}, epoch=epoch+1)
                p_str += ' Eval Loss: {:.4f}'.format(np.mean(losses))

            print(p_str)


def training(exp_names, run_names):

    for exp_name, run_name in zip(exp_names, run_names):
        trainer = Trainer(exp_name=exp_name, run_name=run_name)
        trainer.fit()


###--- Evaluation ---###
# This is the function used for evaluation.
# The different implementations of the evaluation metrics can be found in the TorchGadgets package
# The structure of this function is close to the training function.



###--- Hyperparameter Tuning ---###
# These are the functions used for conducting Optuna studies.


def hyperparameter_tuning(exp_name, run_name):
    print("Hyperparameter tuning not implemented...")
    return




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
    