import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

class ImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args: csv file of train, test, validation with BNPP

        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 0]
        bnpp = self.img_labels.iloc[idx,1]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'Patient': label, 'bnpp_log': bnpp}
        return sample

class ImageSubset(Dataset):
    def __init__(self, csv_file, data, img_dir, transform=None):
        """
        Args: csv file of train, test, validation with BNPP

        """
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.data[idx][0]+'.jpg')
        image = read_image(img_path)
        label = self.data[idx][0]
        bnpp = self.img_labels[self.img_labels['unique_key']==self.data[idx][0]]['bnpp_value_log'].values[0]
        if self.transform:
            image = self.transform(image)
        sample = {'image': image, 'Patient': label, 'bnpp_log': bnpp}
        return sample