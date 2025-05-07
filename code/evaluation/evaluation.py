import torch
import random
from utils.utils import show_batch, show_channel_batch

PALETTE = {
    0: (128, 64, 128),
    1: (244, 35, 232),
    2: (70, 70, 70), 
    3: (102, 102, 156),
    4: (190, 153, 153),
    5: (153, 153, 153),
    6: (250, 170, 30),
    7: (220, 220, 0),
    8: (76, 153, 0),
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
    255: (255, 153, 51)
}


def evaluate(model, dataloader, device):

    model = model.to(device)
    model.eval()

    with torch.no_grad():

        imgs = []      
        preds = []
        targets = []
        for img, target in dataloader:

            img = img.to(device)
            target = target.to(device)
            out = model(img)['out']
            pred = torch.argmax(out, dim = 1) # Out is 4D tensor (bacth_size, num_classes, h, w) -> I want to select the higher class for each pixel

            imgs.append(img.cpu())
            preds.append(pred.cpu())
            targets.append(target.cpu())

        ran_ind = random.randint(0, len(dataloader) - 1)

        batch = (imgs[ran_ind], preds[ran_ind], targets[ran_ind])
        show_batch(batch = batch, palette = PALETTE)
        

        
import torch.nn.functional as F

def evaluate_obstacle_rec_model(model, dataloader, device, channel, treshold):

    model = model.to(device)
    model.eval()

    with torch.no_grad():

        imgs = []      
        preds = []
        targets = []

        for img, target in dataloader:

            img = img.to(device)
            target = target.to(device)

            out = model(img)['out']  # logits
            pred = torch.sigmoid(out)  # normalizza tra 0 e 1

            imgs.append(img.cpu())
            preds.append(pred.cpu())
            targets.append(target.cpu())

        ran_ind = random.randint(0, len(imgs) - 1)

        batch = (imgs[ran_ind], preds[ran_ind], targets[ran_ind])
        show_channel_batch(batch=batch, channel=channel, treshold=treshold)
