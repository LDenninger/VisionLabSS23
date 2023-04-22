import torchvision.datasets as datasets
import torchvision.transforms as transforms

import random
import numpy as np
import torch

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def load_svhn_dataset():
    train_data = datasets.SVHN(root='./data', split='train', transform=transforms.ToTensor(), download=True)
    test_data = datasets.SVHN(root='./data', split='test', transform=transforms.ToTensor(), download=True)
    return train_data, test_data

def load_mnist_dataset():
    train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    return train_data, test_data
