import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
import PIL.Image as Image
from typing import Any, Callable, Optional, Union
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

OBJECTS = [5, 11, 12, 13, 14, 15, 16, 17, 18]

class CityScapes(Cityscapes):
    def __getitem__(self, index):
        
        image = Image.open(self.images[index]).convert("RGB")

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i]) 
            else:
                target = Image.open(self.targets[index][i])  # type: ignore[assignment]

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]  # type: ignore[assignment]

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            target, ignore_mask = to_one_hot_plus_one(target=target)
            
        return image, target, ignore_mask
    
def merge_and_remove(tensor):

    assert tensor.shape[0] > 19

    tensor[19] = torch.logical_or(tensor[19], tensor[20]).to(tensor.dtype)

    tensor = torch.cat((tensor[:20], tensor[21:]), dim=0)

    return tensor

def to_one_hot_plus_one(target):

    target = target.squeeze()
    ignore_mask = (target != 19).float().unsqueeze(0)
    one_hot = torch.nn.functional.one_hot(target, num_classes = 21).permute(2, 0, 1)
    one_hot[19] = 0

    for channel in OBJECTS:

        pos = one_hot[channel] == 1
        one_hot[19][pos] = 1
    
    one_hot = merge_and_remove(one_hot)


    return one_hot, ignore_mask
