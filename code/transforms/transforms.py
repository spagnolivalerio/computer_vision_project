import torch
from torchvision import transforms as T
import numpy as np

CITYSCAPES_ID_TO_TRAINID = {
    0: 6, 1: 6, 2: 6, 3: 6, 4: 6, 5: 6, 6: 6, 7: 0,
    8: 0, 9: 6, 10: 6, 11: 1, 12: 1, 13: 1, 14: 6, 15: 6,
    16: 6, 17: 2, 18: 6, 19: 2, 20: 2, 21: 3, 22: 3, 23: 3,
    24: 4, 25: 4, 26: 5, 27: 5, 28: 5, 29: 6, 30: 6,
    31: 5, 32: 5, 33: 5, 100: 6, 101: 2
}

CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255, 7: 0,
    8: 1, 9: 255, 10: 255, 11: 2, 12: 3, 13: 4, 14: 255, 15: 255,
    16: 255, 17: 5, 18: 255, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18, 100: 19, 101: 20
}

def map_to_train_ids(mask):
    mask = np.array(mask)
    mapped = np.full_like(mask, 6)
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
    T.Lambda(map_to_train_ids) # Here there is the trasformation to a tensor
])