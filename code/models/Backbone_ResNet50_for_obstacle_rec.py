import torchvision
import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights = weights)
model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 20, kernel_size=1)
model = model.cuda()

def custom_bce_loss(logits, targets, ignore_index=19, reduction='mean'):

    mask = targets != ignore_index 

    loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask].float(), reduction=reduction)

    return loss

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
