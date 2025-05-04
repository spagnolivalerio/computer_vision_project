from datasets import cityscapes as ce
from transforms import transforms as t
# Directory delle immagini e delle maschere
images_dir = "../data/images/train/bochum"
masks_dir = "../data/masks/train/bochum"
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Crea dataset e dataloader
dataset = ce.SemanticSegmentationDataset(images_dir, masks_dir, img_transform=t.img_transform, masks_transform=t.mask_transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Itera sul dataloader
for images, masks in dataloader:
    print("Batch immagini shape:", images.shape)  # es. [B, 3, 512, 1024]
    print("Batch maschere shape:", masks.shape)   # es. [B, 512, 1024]
    
    # Visualizza una immagine e maschera dal batch
    img = images[0].permute(1, 2, 0).numpy()  # da [C,H,W] a [H,W,C]
    img = (img * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.406]  # de-normalizza
    img = img.clip(0, 1)

    mask = masks[0].numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Immagine")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title("Maschera")
    plt.axis('off')

    plt.show()
    break  # esci dopo il primo batch
