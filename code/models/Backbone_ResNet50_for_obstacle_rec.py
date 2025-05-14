import torchvision
import torch.nn as nn
import torch
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights = weights)
model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 20, kernel_size=1)
model = model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(), 
    lr=0.01, 
    momentum=0.9, 
    weight_decay=1e-4
)
