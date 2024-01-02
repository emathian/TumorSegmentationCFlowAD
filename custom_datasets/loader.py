import os
from PIL import Image
import numpy as np
import torch
from torchvision.io import read_video, write_jpeg
from torch.utils.data import Dataset
from torchvision import transforms as T

__author__ = "Emilie Mathian"

class TumorNormalDataset(Dataset):
    def __init__(self, c, is_train=True):
        """Tumour segmentation data loader. This data loader is based on a file containing a list of paths to the tiles contained in the training or inference set.
        See: c.list_file_train and c.list_file_test"""
        # Action:
        # is_train ==  True > train CFlow on 'defect-free' images
        # is_test ==  True > inference mode
        # if infer_train == True > Infer train set
        # else > Infer test set
        
        self.is_train = is_train
        self.infer_train = c.infer_train

        # Tiles size
        self.cropsize = c.crp_size
        # Load dataset
        self.list_file_test = c.list_file_test
        self.list_file_train = c.list_file_train
        self.x, self.y, self.mask  = self.load_dataset_folder()

        # data augmentation
        if is_train:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                T.RandomRotation(5),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # test:
        else:
            self.transform_x = T.Compose([
                T.Resize(c.img_size, Image.LANCZOS),
                T.CenterCrop(c.crp_size),
                T.ToTensor()])
        # mask => Useless if any mask are defined
        self.transform_mask = T.Compose([
            T.Resize(c.img_size, Image.NEAREST),
            T.CenterCrop(c.crp_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(c.norm_mean, c.norm_std)])

    def __getitem__(self, idx):
        x, y, mask = self.x[idx], self.y[idx], self.mask[idx]
        # Path to the tile containing patient_id and tile coordinates on the WSI
        filepath = x
        x = Image.open(x)
        x = self.normalize(self.transform_x(x))
        # There are no masks available for the tumour segmentation task. 
        # Hence mask is a null matrix
        mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        return x, y, mask, filepath

    def __len__(self):
        return len(self.x)

    def load_dataset_folder(self):
        # Data set selection

        if self.is_train : ## Training
            phase = 'train' 
            list_file =  self.list_file_train 
        else: ## Inference
            if not self.infer_train: ## Inference of the tiles from testing set
                phase = 'test'
                list_file = self.list_file_test 
            else:
                phase = 'train'  ## Inference of the tiles from training set
                list_file =   self.list_file_train 

        x, y, mask = [], [], []
        img_dir = os.path.join(list_file)
        with open(img_dir, 'r') as f:
            content =  f.readlines()
        files_list = []
        for l in content:
            l =  l.strip()
            if l.find('Tumor') != -1:
                y.append(1)
            else:
                y.append(0)
            files_list.append(l)
        files_list = sorted(files_list)
        x.extend(files_list)
        mask.extend([None] * len(files_list))
        return list(x), list(y), list(mask)