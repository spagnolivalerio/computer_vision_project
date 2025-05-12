import torch.nn.functional as F
import torch

def custom_bce_loss(logits, target, ignore_mask):

    target = target.float()
    ignore_mask = ignore_mask.float() 

    raw_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none') 

    mask = ignore_mask.expand_as(logits)  

    masked_loss = raw_loss * mask

    return masked_loss.sum() / mask.sum().clamp(min=1.0)

def train_one_epoch_obstacle_rec(model, dataloader, optimizer, device):

    model = model.to(device)
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_id, (img, target, ignore_mask) in enumerate(dataloader, start=1):

        ignore_mask = ignore_mask.to(device)
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        out = model(img)['out']
        loss = custom_bce_loss(out, target, ignore_mask)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()

        if batch_id % 10 == 0 or batch_id == num_batches:
            print(f"Batch n. {batch_id}/{num_batches} - Current batch loss: {loss.item():.4f}")
    
    avg_loss = total_loss/num_batches

    print(f"Average loss: {avg_loss}")

    return avg_loss