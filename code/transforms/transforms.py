import torch
from torchvision import transforms as T
import numpy as np

CITYSCAPES_ID_TO_TRAINID = {
    0: 19, 1: 19, 2: 19, 3: 19, 4: 19, 5: 19, 6: 19, 7: 0,
    8: 1, 9: 19, 10: 19, 11: 2, 12: 3, 13: 4, 14: 19, 15: 19,
    16: 19, 17: 5, 18: 19, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 19, 30: 19,
    31: 16, 32: 17, 33: 18
}

def map_to_train_ids(mask):
    mask = np.array(mask)
    mapped = np.full_like(mask, 19)
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