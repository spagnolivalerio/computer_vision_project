import torch
import torchvision
import numpy
from torchvision.datasets import Cityscapes
from transforms import transforms as T
from torch.utils.data import DataLoader
from models import DeepLabV3_ResNet50_model as RN50
from training.training import train_one_epoch
from evaluation.evaluation import evaluate
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5
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

"""
for epoch in range(NUM_EPOCHS):
    loss = train_one_epoch(model, train_dataloader, RN50.optimizer, RN50.criterion, device=device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

torch.save(model.state_dict(), "models/RN50_epoch5.pth")
"""

RN50.mymodel.load_state_dict(torch.load("models/RN50_epoch5.pth"))

evaluate(RN50.mymodel, val_dataloader, device=device)



