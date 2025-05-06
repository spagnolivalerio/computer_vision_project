import os
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import Cityscapes
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import numpy as np
from tqdm import tqdm


CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
    8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
    16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18
}

def map_to_train_ids(mask):
    # Converti PIL → numpy → mappa valori → torch.Tensor
    mask = np.array(mask)
    mapped = np.full_like(mask, 255)
    for k, v in CITYSCAPES_ID_TO_TRAINID.items():
        mapped[mask == k] = v
    return torch.from_numpy(mapped).long()

trasform = T.Compose([
    T.Resize((256, 512)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
])

target_trasform = T.Compose([
    T.Resize((256, 512), interpolation=T.InterpolationMode.NEAREST),
    T.Lambda(map_to_train_ids)
])


root = "../../data"

train_dataset = Cityscapes(root=root, split = 'train', mode = 'fine', target_type= 'semantic', target_transform=target_trasform, transform=trasform)

val_dataset = Cityscapes(root=root, split = 'val', mode = 'fine', target_type= 'semantic', target_transform=target_trasform, transform=trasform)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle = True, num_workers=4)

val_loader = DataLoader(val_dataset, batch_size=2, shuffle = False, num_workers=4)

w = DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights = w)
model.classifier[4] = nn.Conv2d(256, 19, kernel_size=1)
model = model.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=255)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    loop = tqdm(dataloader, desc="Training", leave=False)
    for imgs, targets in loop:
        imgs = imgs.cuda()
        targets = targets.squeeze(1).long().cuda()

        output = model(imgs)['out']
        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def evaluate(model, dataloader):
    model.eval()
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.cuda()
            output = model(imgs)['out']
            preds = torch.argmax(output, dim=1).cpu()
            return imgs.cpu(), preds, targets.squeeze(1)
        
def decode_segmap(segmentation, palette):
    if segmentation.ndim != 2:
        segmentation = segmentation.squeeze()
    seg_img = torch.zeros(3, *segmentation.shape)
    for label, color in palette.items():
        mask = segmentation == label
        for c in range(3):
            seg_img[c][mask] = color[c] / 255.0
    return seg_img


PALETTE = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70),
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    9: (107, 142, 35),
    10: (70, 130, 180),
    11: (220, 20, 60),
    12: (255, 0, 0),
    13: (0, 0, 142),
    14: (0, 0, 70),
    15: (0, 60, 100),
    16: (0, 80, 100),
    17: (0, 0, 230), 
    18: (119, 11, 32),
    255: (224, 224, 224)
}

def show_pred(imgs, preds, targets):
    fig, axs = plt.subplots(len(imgs), 3, figsize=(12, 5 * len(imgs)))
    if len(imgs) == 1:
        axs = [axs]
    for i in range(len(imgs)):
        axs[i][0].imshow(imgs[i].permute(1, 2, 0))
        axs[i][0].set_title("Input")
        axs[i][1].imshow(decode_segmap(preds[i], PALETTE).permute(1, 2, 0))
        axs[i][1].set_title("Output")
        axs[i][2].imshow(decode_segmap(targets[i], PALETTE).permute(1, 2, 0))
        axs[i][2].set_title("GT")
    plt.tight_layout()
    plt.show()

EPOCHS = 5

for epoch in range(EPOCHS):
    loss = train_one_epoch(model, train_loader, optimizer, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss:.4f}")

    imgs, preds, targets = evaluate(model, val_loader)
    show_pred(imgs, preds, targets)