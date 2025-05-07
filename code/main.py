import torch
import torchvision
import numpy
from torchvision.datasets import Cityscapes
from transforms import transforms as T
from torch.utils.data import DataLoader
from models import DeepLabV3_ResNet50_model as RN50
from models import Backbone_ResNet50_for_obstacle_rec as BBRN50
from training.training import train_one_epoch
from evaluation.evaluation import evaluate_obstacle_rec_model
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
from training.training_obstacle_rec import train_one_epoch_obstacle_rec as train
from datasets.Cityscapes_with_objectness import CityScapes as City

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 3
data_root = "../data"

# Creation of cityscapes dataset picking imgs for semantic segmentation task, applying data trasformations.
train_dataset = City(root = data_root, split = "train", mode = "fine", target_type= "semantic", target_transform = T.target_trasform, transform = T.trasform)

val_dataset = Cityscapes(root = data_root, split = "val", mode = "fine", target_type= "semantic", target_transform = T.target_trasform, transform = T.trasform)

#test_dataset = Cityscapes(root = data_root, split = "test", mode = "fine", target_type= "semantic", target_transform = T.target_trasform, transform = T.trasform)

# DataLoader creation
train_dataloader = DataLoader(train_dataset, batch_size = 8, shuffle = True, num_workers = 4)

val_dataloader = DataLoader(val_dataset, batch_size = 8, shuffle = True, num_workers = 4)

#test_dataloader = DataLoader(test_dataset, batch_size = 8, shuffle = True, num_workers = 4)


"""
for epoch in range(NUM_EPOCHS):
    loss = train_one_epoch(model, train_dataloader, RN50.optimizer, RN50.criterion, device=device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

torch.save(model.state_dict(), "models/RN50_epoch5.pth")

RN50.model.load_state_dict(torch.load("models/RN50_epoch5.pth"))

evaluate(RN50.model, val_dataloader, device=device)

"""
model = BBRN50.model
model.load_state_dict(torch.load("models/weights/OBSTACLE_epoch3.pth"))
evaluate_obstacle_rec_model(model, val_dataloader, device=device, channel=19, treshold=0.4)
"""
for epoch in range(NUM_EPOCHS):
    loss = train(model, train_dataloader, BBRN50.optimizer, device=device)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {loss:.4f}")

torch.save(model.state_dict(), "models/OBSTACLE_epoch3.pth")
"""
