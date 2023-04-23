import torch
import torch.nn as nn
import torch.nn.functional as fun
import torchvision as tv
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from pathlib import Path as P

from tqdm import tqdm

from experiments import Logger

from .utils import apply_data_preprocessing

### Training Scripts ###

def run_train_epoch(    model: nn.Module, 
                        dataset: torch.utils.data.DataLoader, 
                        config: dict,  
                        criterion: nn.Module, 
                        optimizer: torch.optim.Optimizer, 
                        device: str="cpu" ):
    """
    Run a single epoch of training.

    Arguments:
        model (nn.Module): Model to train.
        dataset (torch.utils.data.DataLoader): Dataset to train the model on.
        config (dict): Dictionary containing the configuration of the model.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer.
        device (str): Device to run the model on.

    """

    model.train()
    loss_list = []
    if config['verbosity_level'] != 0:
        dataset_iterator = tqdm(enumerate(dataset), total=len(dataset))
    else:
        dataset_iterator = enumerate(dataset)

    for i, (imgs, labels) in dataset_iterator:
        # Prepare inputs
        imgs = imgs.to(device)
        labels = labels.to(device)

        imgs, labels = apply_data_preprocessing(imgs, labels, config)

        imgs, labels = imgs.float(), labels.float()

        # Produce output
        outputs = model(imgs).squeeze().float()

        # Compute loss and backpropagate
        loss = criterion(outputs, labels)

        loss_list.append(loss.cpu().item())

        optimizer.zero_grad()

        loss.backward()

        # Finally update all weights
        optimizer.step()

        if config['verbosity_level'] != 0:
            dataset_iterator.set_description(f'Loss: {loss.item():.4f}')

    return loss_list


