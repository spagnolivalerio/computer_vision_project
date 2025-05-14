import torch.nn.functional as F
import torch

def poly_lr_scheduler(optimizer, base_lr, current_iter, max_iter, power=0.9):

    lr = base_lr * (1 - current_iter / max_iter) ** power

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def get_boundary_mask(target):
    # target shape: (B, C, H, W)
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32, device=target.device).unsqueeze(0).unsqueeze(0)
    
    boundary = F.conv2d(target.float(), kernel, padding=1)
    return (boundary.abs() > 0).float()

def custom_bce_loss(logits, target, ignore_mask, lambda_weight=3.0):
    target = target.float()
    ignore_mask = ignore_mask.float()

    raw_loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')  # (B, C, H, W)

    # Maschera per pixel validi
    mask = ignore_mask.expand_as(raw_loss)
    weights = torch.ones_like(raw_loss)
    weights[:, 19][target[:, 19] == 1] = 1.5
    # ------------------------
    # STANDARD LOSS (BCE masked)
    std_loss = raw_loss * mask * weights
    scalar_std_loss = std_loss.sum() / mask.sum().clamp(min=1.0)

    # ------------------------
    # BOUNDARY LOSS (solo canale object 19)
    boundary_mask = get_boundary_mask(target[:, 19:20])  # (B, 1, H, W)
    boundary_mask_exp = boundary_mask.expand_as(raw_loss)

    # Applica anche ignore_mask ai bordi
    boundary_mask_applied = boundary_mask_exp * mask

    boundary_loss = (raw_loss * boundary_mask_applied).sum() / boundary_mask_applied.sum().clamp(min=1.0)

    # ------------------------
    return scalar_std_loss + lambda_weight * boundary_loss


def train_one_epoch_obstacle_rec(model, dataloader, optimizer, device, epoch, totepochs):

    model = model.to(device)
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_id, (img, target, ignore_mask) in enumerate(dataloader, start=0):

        ignore_mask = ignore_mask.to(device)
        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        current_iter = epoch * len(dataloader) + batch_id
        poly_lr_scheduler(optimizer, base_lr=0.01, current_iter=current_iter, max_iter=totepochs*len(dataloader))

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