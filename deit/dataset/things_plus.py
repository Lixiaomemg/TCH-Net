import os
import cv2
import torch
import numpy as np
from collections import defaultdict
from PIL import Image
from torch.utils.data import Dataset
import clip
import torch.nn as nn
from typing import Optional, Callable, Union, List, Tuple

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class ThingsDataset(Dataset):
    def __init__(self, 
                 root: str,
                 image_paths_file: str,
                 concepts_file: str,
                 category_file: str,
                 is_train: bool = True,
                 split_ratio: float = 0.8, 
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
        self.class_label_list = []  
        self.super_label_list = []  

        # 1. UniqueID -> Fine ID 
        uniqueid_to_fine = {}
        with open(concepts_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t')
            uid_idx = header.index('uniqueID')
            fine_id = 0
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) > uid_idx:
                    uid = parts[uid_idx].strip()
                    uniqueid_to_fine[uid] = fine_id
                    fine_id += 1

        # 2.  Fine ID -> Coarse ID 
        fine_to_coarse = {}
        with open(category_file, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split('\t') 
            num_coarse_classes = len(header) - 2 #  53 / 27
            fine_id = 0
            for line in f:
                parts = line.strip().split('\t')
                coarse_parts = parts[2:] 
                try:
                    coarse_id = coarse_parts.index('1') 
                    if coarse_id >= num_coarse_classes:
                        coarse_id = 0
                except ValueError:
                    coarse_id = 0 
                    
                fine_to_coarse[fine_id] = coarse_id
                fine_id += 1
        paths_by_fine_id = defaultdict(list)
        with open(image_paths_file, 'r', encoding='utf-8') as f:
            for line in f:
                path = line.strip()
                if not path:
                    continue
                folder_name = path.split('/')[1] 
                
                if folder_name in uniqueid_to_fine:
                    f_id = uniqueid_to_fine[folder_name]
                    paths_by_fine_id[f_id].append(path)
        rng = np.random.RandomState(42)
        for f_id, paths in paths_by_fine_id.items():
            paths = sorted(paths) 
            rng.shuffle(paths)
            
            split_idx = int(len(paths) * split_ratio)
            selected_paths = paths[:split_idx] if is_train else paths[split_idx:]
            
            for p in selected_paths:
                self.img_path.append(os.path.join(root, p))
                self.class_label_list.append(f_id)
                self.super_label_list.append(fine_to_coarse[f_id])

        self.targets = self.class_label_list

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