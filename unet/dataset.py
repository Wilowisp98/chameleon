"""
 Implementation based on:
 - https://youtu.be/IHq1t7NxS8k?si=d9dofGF9n96192R8
 - https://youtu.be/HS3Q_90hnDg?si=6BFVv_jLfQLhuA5i
 - https://d2l.ai/chapter_convolutional-modern/batch-norm.html
 - https://www.youtube.com/watch?v=oLvmLJkmXuc
"""

import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

class load(Dataset):
    def __init__(self, data_dir, datasets, original, mask, transform = None, train = True):
        self.data_dir = data_dir
        self.datasets = datasets
        self.transform = transform
        self.suffix = '_train' if train else '_val'
        self.original = original + self.suffix
        self.mask = mask + self.suffix

        self.images = []
        self.masks = []
        for dataset in self.datasets:
            dataset_path = os.path.join(self.data_dir, dataset)
            original_path = os.path.join(dataset_path, self.original)
            mask_path = os.path.join(dataset_path, self.mask)

            for image in os.listdir(original_path):
                self.images.append(os.path.join(original_path, image))
            for mask in os.listdir(mask_path):
                self.masks.append(os.path.join(mask_path, mask)) 

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = np.array(Image.open(self.images[index]).convert("RGB"))
        # L means grayscale
        mask = np.array(Image.open(self.masks[index]).convert("L"), dtype=np.float32)
        # We are using a sigmoid as the last activation function, which indicates the probability of being
        #   a white/black pixel.
        # Since all the points are white OR black the values will be 0.0 or 255.0 so what we will do
        #   is change the 255.0 to 1s so it gets normalized.
        mask[mask == 255.0] = 1.0

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask
