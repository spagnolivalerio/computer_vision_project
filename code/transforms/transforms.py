import cv2
import torch
from torchvision import transforms

def img_transform(image):
    # Resize immagine
    image = cv2.resize(image, (1024, 512))

    # Converti l'immagine in un tensore PyTorch
    image = torch.from_numpy(image.transpose((2, 0, 1)))  # PyTorch works with (C, H, W)
    image = image.float() / 255.0  # Normalizza a [0, 1]

    # Normalizzazione con valori standard ImageNet
    image = transforms.functional.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    return image

def mask_transform(mask):
    # Resize of the mask
    mask = cv2.resize(mask, (1024, 512), interpolation=cv2.INTER_NEAREST)  # Interpolation to make the mask consistent

    # Converti la maschera in un tensore PyTorch
    mask = torch.from_numpy(mask)  # Conversion from NumPy array (default choice of opencv) to PyTorch tensor

    return mask
