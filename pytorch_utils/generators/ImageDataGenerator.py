import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, labels:pd.DataFrame, paths_to_images:pd.DataFrame, paths_prefix:str,
                 augment:bool,):
        self.labels = labels
        self.img_paths = paths_to_images
        self.paths_prefix = paths_prefix

    def __len__(self):
        return self.labels.shape[0]


    def __getitem__(self, idx):




    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label