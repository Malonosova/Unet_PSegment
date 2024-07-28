import pandas as pd
import numpy as np
import cv2
import os 
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class Preprocess(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode
        
        self.size = (256, 256)
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        
    @staticmethod
    def read(names_list, path='Pascal-part'):
        names_list = pd.read_csv(f'{path}/{names_list}', header=None)[0].tolist()
        data = []
        for name in tqdm(names_list):
            img = cv2.imread(f'{path}/JPEGImages/{name}.jpg')
            mask = np.load(f'{path}/gt_masks/{name}.npy')
            data.append([img, mask])
        return data 
            
    def augum_extra(self, img, angle=30):
        transform_ex1 = transforms.Compose([
            transforms.RandomHorizontalFlip(p=1)])
        
#         transform_ex2 = transforms.Compose([
# #             transforms.GaussianNoise(),
#             transforms.RandomRotation(90)])

    
        if self.mode=='train_ex1':
            img = transform_ex1(img)
        else:
            transforms.functional.rotate(img, angle)
            
        return img
            
    
    def __getitem__(self, ind):
        angle = np.random.choice([90, 180, 270]).astype(float)
        img = self.data[ind][0]
        mask = self.data[ind][1]
        
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.size, antialias=True),
            transforms.Normalize(self.mean, self.std),])
        
        transform_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.size, antialias=False)])
        
        mask = transform_mask(mask)*255
        img = transform_img(img)
        if self.mode=='train_ex1' or self.mode=='train_ex2':
            img = self.augum_extra(img, angle)
            mask = self.augum_extra(mask, angle)
        
        return img, mask.round()

    
    def __len__(self):
        return len(self.data)