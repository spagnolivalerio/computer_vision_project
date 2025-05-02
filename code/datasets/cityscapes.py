import os
import cv2
from torch.utils.data import Dataset

class SemanticSegmentationDataset(Dataset):

    def __init__(self, images_dir, masks_dir, img_transform = None, masks_transform = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_filenames = sorted(os.listdir(images_dir))
        self.mask_filenames = sorted(os.listdir(masks_dir))
        self.img_transform = img_transform
        self.masks_transform = masks_transform


    # The method returns the number of images into the dataset
    def __len__(self):
        return len(self.image_filenames)

    # The method returns the image from the dataset with relative mask (it's a tuple)
    def __getitem__(self, id):

        img_name = os.path.join(self.images_dir, self.image_filenames[id])
        mask_name = os.path.join(self.masks_dir, self.mask_filenames[id])

        image = cv2.imread(img_name)
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE) # Load the mask in greyscale

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Conversion from BGR to RGB

        if self.img_transform:
            image = self.img_transform(image)
        if self.masks_transform:
            mask = self.masks_transform(mask)

        return image, mask


