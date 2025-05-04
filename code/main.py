import torch
import torchvision
import numpy
from torchvision.datasets import Cityscapes
from transforms import transforms as T
from torch.utils.data import DataLoader
from models import DeepLabV3_ResNet50_model as RN50

data_root = "../data"

# Creation of cityscapes dataset picking imgs for semantic segmentation task, applying data trasformations.
train_dataset = Cityscapes(root = data_root, split = "train", mode = "fine", target_type= "semantic", target_transform = T.target_trasform, transform = T.trasform)

val_dataset = Cityscapes(root = data_root, split = "val", mode = "fine", target_type= "semantic", target_transform = T.target_trasform, transform = T.trasform)

test_dataset = Cityscapes(root = data_root, split = "test", mode = "fine", target_type= "semantic", target_transform = T.target_trasform, transform = T.trasform)

# DataLoader creation
train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 4)

val_dataloader = DataLoader(val_dataset, batch_size = 8, shuffle = True, num_workers = 4)

test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = True, num_workers = 4)

model = RN50.model

