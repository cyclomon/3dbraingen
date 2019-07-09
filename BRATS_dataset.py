import csv
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from skimage.transform import resize
from nilearn import surface
import nibabel as nib
from skimage import exposure

class BRATSdataset(Dataset):
    def __init__(self, train=True, imgtype = 'flair', severity='HGG',is_flip=False,augmentation=True):
        self.augmentation = augmentation
        if train:
            self.root = '../Training_brats/' + severity
        else:
            self.root = '../Validation_brats'
        self.imgtype = imgtype
        list_img = os.listdir(self.root)
        list_img.sort()
        self.imglist = list_img
        self.is_flip = is_flip
        
    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):

        path = os.path.join(self.root,self.imglist[index])
        
        img = nib.load(os.path.join(path,self.imglist[index]+'_'+self.imgtype+'.nii.gz'))
        gt = nib.load(os.path.join(path,self.imglist[index])+'_'+'seg.nii.gz')

        A = np.zeros((240,240,166))
        G = np.zeros((240,240,166))
        A[:,:,11:] = img.get_data()
        G[:,:,11:] = gt.get_data()
        x=[]
        y=[]
        z=[]
        
        for i in range(240):
            if np.all(A[i,:,:] ==0):
                x.append(i)
            if np.all(A[:,i,:]==0):
                y.append(i)
            if i <155:
                if np.all(A[:,:,i]==0):
                    z.append(i)

        xl,yl,zl = 0,0,0
        xh,yh,zh = 240,240,155
        for xn in x:
            if xn < 120:
                if xn> xl:
                    xl = xn
            else:
                if xn<xh:
                    xh = xn
        for yn in y:
            if yn < 120:
                if yn> yl:
                    yl = yn
            else:
                if yn<yh:
                    yh = yn
        for zn in z:
            if zn < 77:
                if zn> zl:
                    zl = zn
            else:
                if zn<zh:
                    zh = zn

        B = A[xl-10:xh+10,yl-10:yh+10,zl-10:zh+10]
        B = resize(B, (128, 128, 128), mode='constant')
        
        if self.is_flip:
            B = np.swapaxes(B,1,2)
            B = np.flip(B,1)
            B =np.flip(B,2)
        
        sp_size = 64
        img = resize(B, (sp_size,sp_size,sp_size), mode='constant')
        if self.augmentation:
            random_n = torch.rand(1)
            random_i = 0.3*torch.rand(1)[0]+0.7
            if random_n[0] > 0.5:
                img = np.flip(img,0)

        img = 1.0*img
        img = exposure.rescale_intensity(img)
        img = (img-np.min(img))/(np.max(img)-np.min(img))
        img = 2*img-1

        imageout = torch.from_numpy(img).float().view(1,sp_size,sp_size,sp_size)

        return imageout

