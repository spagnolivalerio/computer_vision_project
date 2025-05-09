import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights
import torchvision
import torch.nn as nn
import numpy as np


def get_img(tensor, channel): # The function takes as input a tensor of shape (C, H, W) and return a numpy binary vector related to the specific channel

    if tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    mask = tensor[channel].cpu().numpy()

    h, w = mask.shape
    img = np.zeros((h, w, 3), dtype = np.uint8)

    return img

# Carica il modello con i pesi predefiniti
weights = DeepLabV3_ResNet50_Weights.DEFAULT
model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)

# Modifica l'output per avere 20 classi
model.classifier[-1] = nn.Conv2d(model.classifier[-1].in_channels, 20, kernel_size=1)
model = model.cuda()

# Carica i pesi del modello addestrato
model.load_state_dict(torch.load('../models/weights/OBSTACLE_epoch3.pth'))  # Sostituisci con il percorso corretto
model.eval()  # Imposta il modello in modalità di valutazione

# Definisci le trasformazioni (devono essere le stesse usate per allenare il modello)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converte l'immagine in un tensore
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizzazione
])

# Carica l'immagine
image_path = 'test.png'  # Sostituisci con il percorso dell'immagine
image = Image.open(image_path).convert('RGB')

# Applica le trasformazioni
input_tensor = transform(image).unsqueeze(0).to("cuda")  # Aggiungi una dimensione batch e spostala sulla GPU

# Esegui la previsione
with torch.no_grad():  # Disattiva il calcolo del gradiente per la previsione
    output = model(input_tensor)

# Estrai la previsione (output è un OrderedDict)
output_tensor = output['out']  # Estrai il tensore di previsione
output_tensor = output_tensor.squeeze(0)
# Applica sigmoid per ottenere probabilità
prob = torch.sigmoid(output_tensor)


# Funzione per calcolare il complemento delle probabilità
def anomalies_map(tensor):  # Tensor of shape (C, H, W)
    # Calcola il complemento delle probabilità
    complement = 1.0 - tensor  # Complemento delle probabilità

    # Calcola la "produttoria delle probabilità complementari" per ogni pixel lungo le classi (C)
    product_of_complements = torch.prod(complement, dim=0)  # (H, W), la moltiplicazione lungo la dimensione C

    # Convertiamo la mappa in numpy per la visualizzazione
    anomalies_map = product_of_complements.cpu().numpy()

    return anomalies_map

# Calcola la mappa delle anomalie (complemento delle probabilità)
map = anomalies_map(prob)

# Assicurati che la mappa sia 2D per visualizzarla correttamente
map = map.squeeze()
masked_map = map.copy()
#   masked_map[masked_map <= 0.45] = 0  # oppure np.nan per trasparenza

plt.imshow(masked_map, cmap="jet")
plt.colorbar()
plt.title("Pixel con valore > 0.8")
plt.axis("off")
plt.show()
