import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torch


class MyDataset(Dataset):
    def __init__(self, stride, patch_size, patches):

        self.stride = stride
        self.transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
                                          ])
        self.samples = []
        self.patch_size = patch_size
        self.patches = patches

        for file in os.listdir('images/rgb_images'):
            if os.path.exists(f'images/processed_gt_images/IMG_Mask_{file[-8:]}'):
                with Image.open(os.path.join('images/rgb_images', file)) as image:
                    image = np.array(image)
                    if self.patches is True:
                        for x in range(0, image.shape[0] - self.patch_size, self.stride):
                            for y in range(0, image.shape[1] - self.patch_size, self.stride):
                                self.samples.append([x, y, file])
                    else:
                        self.samples.append(file)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        if self.patches is True:
            x, y, file = self.samples[idx]
        else:
            file = self.samples[idx]

        with Image.open(os.path.join('images/rgb_images', file)) as image:
            image = np.array(image)
            if self.patches is True:
                rgb_patch = image[x : x + self.patch_size, y : y + self.patch_size]
                input_image = Image.fromarray(rgb_patch)
            else:
                input_image = Image.fromarray(image)
            processed_rgb = self.transform(input_image)

        with Image.open(os.path.join('images/processed_gt_images', f'IMG_Mask_{file[-8:]}')) as image_mask:
            image_mask = np.array(image_mask)
            if self.patches is True:
                mask_patch = image_mask[x : x + self.patch_size, y : y + self.patch_size]
                processed_gt = torch.from_numpy(mask_patch).long()
            else:
                processed_gt = torch.from_numpy(image_mask).long()
        return processed_rgb, processed_gt, f'{file[-8:]}'