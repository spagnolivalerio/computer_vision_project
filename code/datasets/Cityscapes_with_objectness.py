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

OBJECTS = [5]

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
        target_filename = os.path.basename(self.targets[index][0])

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            target, ignore_mask = to_one_hot_plus_one(target, target_filename)
            
        return image, target, ignore_mask

def to_one_hot_plus_one(target, file):

    target = target.squeeze()
    ignore_mask = (target != 6).float().unsqueeze(0)
    one_hot = torch.nn.functional.one_hot(target, num_classes = 7).permute(2, 0, 1)
    one_hot = torch.cat((one_hot[:5], one_hot[6:]), dim=0)
    
    return one_hot, ignore_mask
