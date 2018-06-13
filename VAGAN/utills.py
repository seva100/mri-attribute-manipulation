import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os

from skimage import img_as_float
from skimage.transform import resize
from skimage.io import imread, imsave


class mri_data(Dataset):
    def __init__(self, 
                 folder='bids_slices', 
                 metadata='matched_saved_exp.csv', 
                 columns=['anat_ras','ADAS13'],
                 resize_shape = (128,128),
                 DX_dict = {'CN':0,'MCI':1,'Dementia':2},
                 transform=None):
        
        self.meta = pd.read_csv(metadata)[['id']+columns]
        self.folder = folder
        self.resize_shape = resize_shape
        self.DX_dict = DX_dict
        self.transform = transform
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        
        if isinstance(idx, slice):
            imgs, attrs = [], []
            
            if idx.start is None:
                start = 0
            else:
                start = idx.start
                
            if idx.stop is None:
                stop = 0
            else:
                stop = idx.stop
                
            for i in range(start, stop):
                true_path = self.folder +'/' + self.meta.anat_ras[i][54:-7] + '.png'
                img = imread(fname=true_path)
                
                attr = self.meta.iloc[i].drop(['anat_ras','id'])
                attr['DX'] = self.DX_dict[attr['DX']]
                attrs.append((np.array(attr, dtype=np.float32)))
                
                if not self.resize_shape is None:
                    img = resize(img, output_shape=self.resize_shape, mode='reflect', preserve_range=True)
                
                img = img_as_float(img)
                
                if self.transform is not None:
                    img = self.transform(img)
                    
                imgs.append(img)
                
                attr = self.meta.iloc[i].drop(['anat_ras','id'])
                if 'DX' in self.meta.columns.values.tolist():
                    attr['DX'] = self.DX_dict[attr['DX']]
                attrs.append((np.array(attr, dtype=np.float32)))
                
            return np.array(imgs), np.array(attrs)
            
        elif isinstance(idx,int):
            
            true_path = self.folder +'/' + self.meta.anat_ras[idx][54:-7] + '.png'
            img = imread(fname=true_path)

            attr = self.meta.iloc[idx].drop(['anat_ras','id'])
            
            if 'DX' in self.meta.columns.values.tolist():
                    attr['DX'] = self.DX_dict[attr['DX']]
            
            if not self.resize_shape is None:
                img = resize(img, output_shape=self.resize_shape, mode='reflect', preserve_range=True)
            
            img = img_as_float(img)
            
            if self.transform is not None:
                img = self.transform(img)

            return img, np.array(attr, dtype=np.float32)

