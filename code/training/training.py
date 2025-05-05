def train_one_epoch(model, dataloader, optimizer, criterion, device):

    model = model.to(device)
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch_id, (img, target) in enumerate(dataloader, start=1):

        img = img.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        out = model(img)['out']
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        total_loss = total_loss + loss.item()

        if batch_id % 10 == 0 or batch_id == num_batches:
            print(f"Batch n. {batch_id}/{num_batches} - Current batch loss: {loss.item():.4f}")
    
    avg_loss = total_loss/num_batches

    print(f"Average loss: {avg_loss}")

    return avg_loss
