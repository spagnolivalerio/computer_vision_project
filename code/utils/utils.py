import matplotlib.pyplot as plt
import torch

OBJECTS = [5, 11, 12, 13, 14, 15, 16, 17, 18]

def show_segmap(tensor, palette):
    if tensor.ndim != 2:
        tensor = tensor.squeeze()
    seg_img = torch.zeros(3, *tensor.shape, dtype = torch.float32)
    for label, color in palette.items():
        mask = (tensor == label) # Boolean tensor
        for c in range(3):
            seg_img[c][mask] = color[c] / 255.0
    return seg_img

def show_batch(batch, palette): # bacth is a tuple such as bacth = (imgs, preds, targets)
    
    batch_size = batch[0].shape[0]
    imgs = batch[0]
    preds = batch[1]
    targets = batch[2]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axs = plt.subplots(batch_size, 3, figsize = (12, 4 * batch_size))

    for i in range(batch_size):
        
        img = imgs[i].cpu() * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1)

        pred_img =show_segmap(preds[i].cpu(), palette).permute(1, 2, 0)
        target_img = show_segmap(targets[i].cpu(), palette).permute(1, 2, 0)

        axs[i][0].imshow(img)
        axs[i][0].set_title("Input")
        axs[i][1].imshow(pred_img)
        axs[i][1].set_title("Prediction")
        axs[i][2].imshow(target_img)
        axs[i][2].set_title("Ground Truth")

        for j in range(3):
            axs[i][j].axis("off")

    plt.tight_layout()
    plt.show()

def show_batch2(batch, palette):
    """
    Funzione per visualizzare un batch di immagini di input, previsione e ground truth.
    """
    batch_size = batch[0].shape[0]
    imgs = batch[0]
    preds = batch[1]
    targets = batch[2]
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    fig, axs = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    for i in range(batch_size):
        
        # De-normalizzazione dell'immagine
        img = imgs[i].cpu() * std + mean
        img = img.permute(1, 2, 0).clamp(0, 1)

        # Soglia a 0.5 per ottenere la previsione binaria (classe 19)
        pred_img = (preds[i][19] > 0.8).float()  # Seleziona il canale 19 e applichiamo la soglia

        # Visualizziamo le immagini
        axs[i][0].imshow(img)
        axs[i][0].set_title("Input")
        axs[i][1].imshow(pred_img)
        axs[i][1].set_title("Prediction (class 19)")

        for j in range(3):
            axs[i][j].axis("off")

    plt.tight_layout()
    plt.show()