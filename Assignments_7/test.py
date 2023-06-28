import torch
import torchvision as tv

train_ds = tv.datasets.SVHN(root='./data', split='train', download=True,)
train_ds = tv.datasets.SVHN(root='./data', split='test', download=True)
