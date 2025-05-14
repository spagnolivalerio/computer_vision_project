import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision
import torch.nn as nn
import numpy as np

# === PARAMETRI CHIAVE ===
OBJECT_CHANNEL_INDEX = 6  # Cambia se il canale oggetto è altrove
NUM_CLASSES = 7           # Numero di classi (escluso ignore)
WEIGHTS_PATH = '../models/weights/low_complexity-SGD-boundary-aware-bs=4-weights=2.0-4.pth'
IMAGE_PATH = 'test.png'

# === MODELLO ===
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval().cuda()

# === TRASFORMAZIONE ===
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# === CARICAMENTO IMMAGINE ===
image = Image.open(IMAGE_PATH).convert('RGB')
input_tensor = transform(image).unsqueeze(0).to("cuda")

# === PREDIZIONE ===
with torch.no_grad():
    output = model(input_tensor)['out'].squeeze(0)
    prob = torch.sigmoid(output)  # shape: (C, H, W)

# === FUNZIONE: mappa UO ===
def anomalies_map(prob):
    obj = prob[OBJECT_CHANNEL_INDEX]        # oggetto
    known = prob[:6]
    unknown = torch.prod(1.0 - known, dim=0)
    uo_map = obj * unknown
    return uo_map.cpu().numpy()

# === VISUALIZZAZIONE ===
uo_map = anomalies_map(prob)
uo_map = uo_map.squeeze()

# Visualizza solo dove il valore è significativo
mask = uo_map > 0.5 * uo_map.max()
masked_map = np.zeros_like(uo_map)
masked_map[mask] = uo_map[mask]

plt.imshow(uo_map, cmap="jet")
plt.title("Unknown Objectness Map")
plt.colorbar()
plt.axis("off")
plt.show()

# (Opzionale) Visualizza anche il canale object puro
plt.imshow(prob[OBJECT_CHANNEL_INDEX].cpu(), cmap="jet")
plt.title("Object Channel Prediction")
plt.colorbar()
plt.axis("off")
plt.show()
