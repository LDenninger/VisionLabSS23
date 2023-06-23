import torch
import os 
from pathlib import Path as P

import torchgadgets as tg

import utils

import json

from tqdm import tqdm

DATASET_NAME = 'food101'
SAVE_NAME = 'food101_processed'


preprocessing = [
                    {
                        "type": "toTensor",
                        "train": True,
                        "eval": True
                    },
                    {
                        "type": "dynamic_center_crop",
                        "train": True,
                        "eval": True
                    },
                    {
                        "type": "resize",
                        "size": [
                            64,
                            64
                        ],
                        "train": True,
                        "eval": True
                    }
                ]

if __name__=='__main__':

    # Data directory as defined by the PyTorch dataset
    DATA_DIR = P(os.getcwd()) / 'data' / DATASET_NAME

    # Create new directory for the processed data
    SAVE_DIR = P(os.getcwd()) / 'data' / SAVE_NAME
    TRAIN_DIR = P(os.getcwd()) / 'data' / SAVE_NAME / 'train'
    TEST_DIR = P(os.getcwd()) / 'data' / SAVE_NAME / 'test'

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    if not os.path.exists(TRAIN_DIR):
        os.makedirs(TRAIN_DIR)
    if not os.path.exists(TEST_DIR):
        os.makedirs(TEST_DIR)


    data = tg.data.load_dataset(DATASET_NAME)
    train_dataset = data['train_dataset']
    test_dataset = data['test_dataset']
    train_dataset = tg.data.ImageDataset(dataset=train_dataset, transforms=preprocessing)
    test_dataset = tg.data.ImageDataset(dataset=test_dataset, transforms=preprocessing, train_set=False)

    labels = []

    ##-- Extract Train Dataset --##

    progress_bar = tqdm(enumerate(train_dataset), total=len(train_dataset))

    for i, (img, label) in progress_bar:

        labels.append(label)
        torch.save(img, str(P(TRAIN_DIR) / f'img_{str(i).zfill(6)}.pt'))

    
    with open(P(TRAIN_DIR) / 'labels.json', 'w') as f:
        json.dump(labels, f)

    ##-- Extract Test Dataset --##

    labels = []

    progress_bar = tqdm(enumerate(test_dataset), total=len(test_dataset))

    for i, (img, label) in progress_bar:

        labels.append(label)
        torch.save(img, str(P(TEST_DIR) / f'img_{str(i).zfill(6)}.pt'))

    
    with open(P(TEST_DIR) / 'labels.json', 'w') as f:
        json.dump(labels, f)
    
    
