import os
from pathlib import Path as P
from PIL import Image
import torchvision as tv
import torch
from tqdm import tqdm
import json

DATASET_PATH = P("Market-1501-processed")
SAVE_TRAIN_DIR = DATASET_PATH / 'train'
SAVE_TEST_DIR = DATASET_PATH / 'test'

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)
if not os.path.exists(SAVE_TRAIN_DIR):
    os.makedirs(SAVE_TRAIN_DIR)
if not os.path.exists(SAVE_TEST_DIR):
    os.makedirs(SAVE_TEST_DIR)

ORIGINAL_PATH = "Market-1501-v15.09.15"
TRAIN_DIR = ORIGINAL_PATH / 'bounding_box_train'
TEST_DIR = ORIGINAL_PATH / 'bounding_box_test'

meta_dir = {}

img_paths = os.listdir(TRAIN_DIR )
img_paths = [el for el in img_paths if os.path.splitext(el)[1] == '.jpg']

tensor_converter = tv.transforms.ToTensor()

progress_bar = tqdm(enumerate(img_paths),total=len(img_paths))

for img_path in progress_bar:
    p_id = img_path[:4]
    file_name = img_path[5:-5]
    if not os.path.exists(SAVE_TRAIN_DIR / f"person_{p_id}"):
        os.makedirs(SAVE_TRAIN_DIR / f"person_{p_id}")
    if p_id not in meta_dir.keys():
        meta_dir[p_id] = 0
    meta_dir[p_id] = meta_dir[p_id] + 1
    img = Image.open(TRAIN_DIR / img_path)
    img = tensor_converter(img)
    torch.save(img, SAVE_TRAIN_DIR / f"person_{p_id}" / f"{file_name}.pt")
meta_dir["length"] = len(img_paths)
json.dump(meta_dir, open(SAVE_TRAIN_DIR / "meta.json"), indent=4)

meta_dir = {}

progress_bar = tqdm(enumerate(img_paths),total=len(img_paths))

img_paths = os.listdir(TEST_DIR)
img_paths = [el for el in img_paths if os.path.splitext(el)[1] == '.jpg']

for img_path in progress_bar:
    p_id = img_path[:4]
    file_name = img_path[5:-5]
    if not os.path.exists(SAVE_TEST_DIR / f"person_{p_id}"):
        os.makedirs(SAVE_TEST_DIR / f"person_{p_id}")
    if p_id not in meta_dir.keys():
        meta_dir[p_id] = 0
    meta_dir[p_id] = meta_dir[p_id] + 1
    img = Image.open(TRAIN_DIR / img_path)
    img = tensor_converter(img)
    torch.save(img, SAVE_TEST_DIR / f"person_{p_id}" / f"{file_name}.pt")

meta_dir["length"] = len(img_paths)
json.dump(meta_dir, open(SAVE_TRAIN_DIR / "meta.json"), indent=4)
