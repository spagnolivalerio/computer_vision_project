import numpy as np
import torch
import matplotlib.pyplot as plt
## torchvision related imports
import torchvision.transforms.functional as F
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.utils import make_grid
## models and transforms
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.segmentation import fcn_resnet50
from torchvision.models.segmentation import FCN_ResNet50_Weights

## utilities for multiple images
def img_show(images):
    if not isinstance(images, list):
    ## generalise cast images to list
        images = [images]
    fig, axis = plt.subplots(ncols=len(images), squeeze=False)
    for i, image in enumerate(images):
        image = image.detach()
        image = F.to_pil_image(image)
        axis[0, i].imshow(np.asarray(image))
        axis[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    plt.tight_layout()
    plt.show()  

            ## get an image on which segmentation needs to be done
img1 = read_image("car.jpg")


batch_imgs = torch.stack([img1])
batch_torch = convert_image_dtype(batch_imgs,dtype=torch.float)

model = fcn_resnet50(weights=FCN_ResNet50_Weights.DEFAULT, progress=False)
## switching on eval mode
model = model.eval()
# standard normalizing based on train config
normalized_batch_torch = F.normalize(batch_torch, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
result = model(normalized_batch_torch)['out']

classes = [
'__background__', 'aeroplane', 'bicycle', 'bird', 'boat',
'bottle', 'bus','car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
'horse', 'motorbike',
'person', 'pottedplant', 'sheep', 'sofa', 'train',
'tvmonitor'
]
class_to_idx = {cls: idx for (idx, cls) in enumerate(classes)}
normalized_out_masks = torch.nn.functional.softmax(result, dim=1)

car_mask = [
normalized_out_masks[img_idx, class_to_idx[cls]]
for img_idx in range(batch_torch.shape[0])
for cls in ('car', 'pottedplant','bus')]
img_show(car_mask[0])