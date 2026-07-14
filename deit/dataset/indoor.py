import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import clip
import torch.nn as nn
from typing import Optional, Callable, Union, List, Tuple

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


INDOOR67_SUPER_DICT = {
    # 0: Store (12)
    'bakery': 0, 'bookstore': 0, 'clothingstore': 0, 'deli': 0, 'florists': 0, 
    'grocerystore': 0, 'hairsalon': 0, 'jewelryshop': 0, 'mall': 0, 'shoeshop': 0, 
    'toystore': 0, 'videostore': 0,
    
    # 1: Home (9)
    'bathroom': 1, 'bedroom': 1, 'children_room': 1, 'closet': 1, 'dining_room': 1, 
    'kitchen': 1, 'livingroom': 1, 'nursery': 1, 'pantry': 1,
    
    # 2: Public Spaces (21)
    'airport_inside': 2, 'artstudio': 2, 'auditorium': 2, 'cloister': 2, 'corridor': 2, 
    'elevator': 2, 'escalator': 2, 'greenhouse': 2, 'hospitalroom': 2, 'inside_bus': 2, 
    'inside_subway': 2, 'laundromat': 2, 'library': 2, 'lobby': 2, 'museum': 2, 
    'operating_room': 2, 'prisoncell': 2, 'stairscase': 2, 'subway': 2, 'trainstation': 2, 
    'waitingroom': 2,
    
    # 3: Leisure (15)
    'bar': 3, 'bowling': 3, 'buffet': 3, 'casino': 3, 'church_inside': 3, 'cinema': 3, 
    'concert_hall': 3, 'fastfood_restaurant': 3, 'gameroom': 3, 'gym': 3, 
    'movietheater': 3, 'poolinside': 3, 'restaurant': 3, 'restaurant_kitchen': 3, 
    'winecellar': 3,
    
    # 4: Working Place (10)
    'classroom': 4, 'computerroom': 4, 'dentaloffice': 4, 'laboratory': 4, 
    'meeting_room': 4, 'office': 4, 'studio': 4, 'tv_studio': 4, 'locker_room': 4,
    'kindergarden': 4
}


class Indoor67Dataset(Dataset):
    def __init__(self, 
                 root: str,           # 对应 /data/xiaomeng/dataset/indoorCVPR/
                 txt_file: str,       
                 is_train: bool = True,
                 transform: Optional[Callable] = None,
                 is_hier: bool = True,
                 category: str = 'name',
                 mean: Union[List, Tuple] = IMAGENET_DEFAULT_MEAN,
                 std: Union[List, Tuple] = IMAGENET_DEFAULT_STD,
                 n_segments: int = 256,
                 compactness: float = 10.0,
                 blur_ops: Optional[Callable] = None,
                 scale_factor: float = 1.0):
        
        self.mean = mean
        self.std = std
        self.n_segments = n_segments
        self.compactness = compactness
        self.blur_ops = blur_ops
        self.scale_factor = scale_factor
        self.is_hier = is_hier
        self.category = category
        self.transform = transform

        self.img_path = []
        self.class_label_list = []  # (0-66)
        self.super_label_list = []  # (0-4)

        unique_classes = []
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if not path: continue
                class_name = path.split('/')[0] 
                if class_name not in unique_classes:
                    unique_classes.append(class_name)
        
        unique_classes.sort() 
        name_to_fine = {name: idx for idx, name in enumerate(unique_classes)}

        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if not path: continue
                
                class_name = path.split('/')[0]
                fine_id = name_to_fine[class_name]
                
                coarse_id = INDOOR67_SUPER_DICT.get(class_name, 2) 
                
                full_path = os.path.join(root, 'Images', path) 
                
                self.img_path.append(full_path)
                self.class_label_list.append(fine_id)
                self.super_label_list.append(coarse_id)

        self.targets = self.class_label_list
        self.unique_classes = unique_classes 

    def __len__(self):
        return len(self.class_label_list)

    def __getitem__(self, index):
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        compactness = self.compactness
        blur_ops = self.blur_ops
        n_segments = self.n_segments
        scale_factor = self.scale_factor
        
        if isinstance(sample, (list, tuple)):
            if not isinstance(compactness, (list, tuple)): compactness = [compactness] * len(sample)
            if not isinstance(n_segments, (list, tuple)): n_segments = [n_segments] * len(sample)
            if not isinstance(blur_ops, (list, tuple)): blur_ops = [blur_ops] * len(sample)
            if not isinstance(scale_factor, (list, tuple)): scale_factor = [scale_factor] * len(sample)

            segments = []
            for samp, comp, n_seg, blur_op, scale in zip(sample, compactness, n_segments, blur_ops, scale_factor):
                if blur_op is not None: samp = blur_op(samp)
                samp = (samp.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
                samp = (samp * 255).astype(np.uint8)
                samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
                seeds = cv2.ximgproc.createSuperpixelSEEDS(
                    samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                    histogram_bins=5, double_step=False)
                seeds.iterate(samp, num_iterations=15)
                segments.append(torch.LongTensor(seeds.getLabels()))
        else:
            if blur_ops is not None: sample = blur_ops(sample)
            samp = (sample.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
            samp = (samp * 255).astype(np.uint8)
            samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2,
                histogram_bins=5, double_step=False)
            seeds.iterate(samp, num_iterations=15)
            segments = torch.LongTensor(seeds.getLabels())

        if self.is_hier:
            return sample, segments, self.class_label_list[index], self.super_label_list[index]
        else:
            if self.category == 'name':
                return sample, segments, self.class_label_list[index]    
            else:
                return sample, segments, self.super_label_list[index]