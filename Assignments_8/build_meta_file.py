import os
from pathlib import Path as P
from PIL import Image
import torchvision as tv
import torch
from tqdm import tqdm
import json

ORIGINAL_PATH = P("Market-1501-v15.09.15")
TRAIN_DIR = ORIGINAL_PATH / 'bounding_box_train'
TEST_DIR = ORIGINAL_PATH / 'query'

meta_dir = {}
img_paths = os.listdir(TRAIN_DIR )
img_paths = [el for el in img_paths if os.path.splitext(el)[1] == '.jpg']

for img_path in tqdm(img_paths):

    p_id = img_path[:4]
    if p_id not in meta_dir.keys():
        meta_dir[p_id] = []
    meta_dir[p_id].append(img_path)


with open(ORIGINAL_PATH / 'train_meta_dir.json', 'w') as f:
    json.dump(meta_dir, f, indent=4)

meta_dir = {}
img_paths = os.listdir(TEST_DIR)
img_paths = [el for el in img_paths if os.path.splitext(el)[1] == '.jpg']

for img_path in tqdm(img_paths):

    p_id = img_path[:4]
    if p_id not in meta_dir.keys():
        meta_dir[p_id] = []
    meta_dir[p_id].append(img_path)


with open(ORIGINAL_PATH / 'test_meta_dir.json', 'w') as f:
    json.dump(meta_dir, f, indent=4)