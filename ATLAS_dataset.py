import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
from skimage.transform import resize
import nibabel as nib
from skimage import exposure

class ATLASdataset(Dataset):
    def __init__(self,augmentation=True):
        list_path = []
        for i in range(9):
            root = '../ATLAS_R1.1/Site'+str(i+1)
        
            list_img = os.listdir(root)
            for s in range(len(list_img)):
                list_path.append(os.path.join(root,list_img[s]))

        list_path.sort()
        self.augmentation= augmentation
        self.imglist = list_path
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        path = os.path.join(self.imglist[index],'t01')
        tempimg = nib.load(os.path.join(path,'T1w_p.nii'))
        B = np.flip(tempimg.get_data(),1)
        sp_size = 64
        img = resize(B, (sp_size,sp_size,sp_size), mode='constant')
        img = 1.0*img
        img = (img-np.min(img))/(np.max(img)-np.min(img))

        if self.augmentation:
            random_n = torch.rand(1)
            if random_n[0] > 0.5:
                img = np.flip(img,0)
                
        img = np.ascontiguousarray(img,dtype=np.float32)

        imageout = torch.from_numpy(img).float().view(1,sp_size,sp_size,sp_size)
        imageout = 2*imageout-1

        return imageout 


