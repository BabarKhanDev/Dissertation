# Import the required libraries
import torch
import os
import torchvision.datasets as datasets
import torchvision
from torch.utils.data import Dataset, DataLoader, random_split
import torch.utils.data
from torchvision import transforms
from torch.nn.functional import interpolate
import torch.nn as nn
from torch.autograd import Variable
import math
import numpy as npP
from natsort import natsorted
from PIL import Image
import random
import pickle
import pandas as pd
import shutil

# Create datasets and dataloaders
def split(dataset_dir = 'VISCHEMA_PLUS/', image_dir = 'images/', label_dir = 'vms/', train = True, transform = None):
        
        if train:
            train_csv = pd.read_csv(f"{dataset_dir}viscplus_train.csv", header = None)
            all_images = train_csv[0].values.tolist()
            dest_dir = "train/"
        else:
            val_csv = pd.read_csv(f"{dataset_dir}viscplus_val.csv" , header = None)
            all_images = val_csv[0].values.tolist()
            dest_dir = "val/"

        for image in all_images:
            shutil.copy(f'{dataset_dir}{image_dir}{image}', f'split_dataset/{dest_dir}{image_dir}{image}')
            shutil.copy(f'{dataset_dir}{label_dir}{image}', f'split_dataset/{dest_dir}{label_dir}{image}')

train_dataset = split(train=True)
val_dataset   = split(train=False)