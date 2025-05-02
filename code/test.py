from datasets import cityscapes as ce
from transforms import transforms as t
# Directory delle immagini e delle maschere
images_dir = "../data/images/test/berlin"
masks_dir = "../data/masks/test/berlin"

# Crea l'istanza del dataset
dataset = ce.SemanticSegmentationDataset(
    images_dir=images_dir,
    masks_dir=masks_dir,
    img_transform=t.img_transform,
    masks_transform=t.mask_transform
)
# Testa il dataset
import matplotlib.pyplot as plt

# Ottieni un'immagine e la maschera dal dataset
image, mask = dataset[0]  # Ottieni la prima immagine e la sua maschera

# Stampa le forme
print(f"Image shape: {image.shape}")
print(f"Mask shape: {mask.shape}")

# Visualizza l'immagine
plt.imshow(image.permute(1, 2, 0))  # Cambia l'ordine dei canali per imshow
plt.title('Image')
plt.show()

# Visualizza la maschera
plt.imshow(mask, cmap='gray')  # La maschera è un'immagine 2D
plt.title('Mask')
plt.show()