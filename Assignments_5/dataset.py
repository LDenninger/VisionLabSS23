import torch
import os
from pathlib import Path as P
import torchgadgets as tg



class KTHActioNDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_name: str, split: str, transforms=None):
        assert split in ['train', 'validation', 'test'], f'Split {split} does not exists'
        self.data_path = P(os.getcwd()) / 'data' / dataset_name / split
        self.split = split
        self.data_augmentor = None

        self.labels = torch.load(str(self.data_path / 'labels.pt'))

        if transforms is not None:
            self.data_augmentor = tg.data.ImageDataAugmentor(transforms)

    def __getitem__(self, index):

        seq_path = self.data_path / f'sequence_{str(index).zfill(5)}.pt'
        img_seq = torch.load(seq_path)
        label = self.labels[index]
        if self.data_augmentor is not None:
            img_seq, label = self.data_augmentor((img_seq, label), True if self.split == 'train' else False)
        return img_seq, label
    
    def __len__(self):
        return self.labels.shape[0]