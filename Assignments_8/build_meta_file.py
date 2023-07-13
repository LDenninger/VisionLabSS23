import os
from pathlib import Path as P
from PIL import Image
import torchvision as tv
import torch
from tqdm import tqdm
import json

ORIGINAL_PATH = P("data/Market-1501")
TRAIN_DIR = ORIGINAL_PATH / 'bounding_box_train'
TEST_DIR = ORIGINAL_PATH / 'bounding_box_test'

meta_dir = {}
img_paths = os.listdir(TRAIN_DIR )
img_paths = [el for el in img_paths if os.path.splitext(el)[1] == '.jpg']
train_length = len(img_paths)

for i, img_path in tqdm(enumerate(img_paths)):

    p_id = int(img_path.split('_')[0])
    if p_id not in meta_dir.keys():
        meta_dir[p_id] = {}
    meta_dir[p_id][i] = img_path


with open(ORIGINAL_PATH / 'train_meta_dir.json', 'w') as f:
    json.dump(meta_dir, f, indent=4)

with open(ORIGINAL_PATH / 'train_img_paths.json', 'w') as f:
    json.dump(img_paths, f, indent=4)

meta_dir = {}
img_paths = os.listdir(TEST_DIR)
img_paths = [el for el in img_paths if (os.path.splitext(el)[1] == '.jpg' and int(el.split('_')[0]) not in [-1, 0])]
test_length = len(img_paths)

for i, img_path in tqdm(enumerate(img_paths)):

    p_id = int(img_path.split('_')[0])
    
    if p_id not in meta_dir.keys():
        meta_dir[p_id] = {}
    meta_dir[p_id][i] = img_path



with open(ORIGINAL_PATH / 'test_meta_dir.json', 'w') as f:
    json.dump(meta_dir, f, indent=4)

with open(ORIGINAL_PATH / 'test_img_paths.json', 'w') as f:
    json.dump(img_paths, f, indent=4)

print("Meta data creation finished!")
print(f' Training length: {train_length}')
print(f' Test length: {test_length}')