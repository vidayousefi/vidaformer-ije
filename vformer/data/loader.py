# -*- coding: utf-8 -*-
import os
from os.path import isfile, join
from PIL import Image
from torch.utils.data import Dataset


class Div2kDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = sorted([f for f in os.listdir(img_dir) if isfile(join(img_dir, f))])
        # self.img_paths = self.img_paths[:20]
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])
        image = self._load_image(img_path)
        if self.transform:
            image = self.transform(image)
        return image

    def _load_image(self, img_path):
        image = Image.open(img_path)
        image = image.convert("RGB")
        return image
