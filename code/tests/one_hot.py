import torch

labels = torch.tensor([[2, 1, 2], [1, 0, 1]])

one_hot = torch.nn.functional.one_hot(labels, num_classes=4)
one_hot = one_hot.permute(2, 0, 1)
print(one_hot)