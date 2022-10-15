import os
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

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
    def __init__(self, csv_file, h5py_file, transform=None):
        """
        Args: 
        csv_file: pandas dataframe with patientID and bnpp value
        h5py_file: h5py file with images and patientID
        """
        self.transform = transform

        #merging the csv_file with the h5py_file
        dict = {}
        for i, key in enumerate(list(h5py_file.keys())):
            #creating dataframe for images, converting images to PIL first
            dict[i] = [key, Image.fromarray(h5py_file[key][:])]
        df = pd.DataFrame(dict).T
        df.columns = ['patientID', 'image']

        csv = csv_file[['unique_key','bnpp_value_log']]
        self.data = df.merge(csv, how='inner',left_on='patientID',right_on='unique_key')

    def __len__(self):
        return len(self.data['patientID'])

    def __getitem__(self, idx):
        #img_path = os.path.join(self.img_dir, self.data[idx][0]+'.jpg')
        #image = read_image(img_path)
        image = self.data.loc[idx,'image']
        #print(image)
        #patient = self.data.loc[idx,'patientID']
        #print(patient)
        bnpp = self.data.loc[idx,'bnpp_value_log']
        #print(bnpp)
        #label = self.data[idx][0]
        #print(label)
        #bnpp = self.img_labels[self.img_labels['unique_key']==self.data[idx][0]]['bnpp_value_log'].values[0]
        #print(bnpp)
        if self.transform:
            image = self.transform(image)
            #print('4')
        #sample = {'image': image, 'Patient': label, 'bnpp_log': bnpp}
        return image, bnpp