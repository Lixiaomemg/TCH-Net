import os
import cv2
import torch
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import clip
import torch.nn as nn
from collections import defaultdict
from typing import Optional, Callable, Union, List, Tuple, Dict, Set

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def get_miniplaces_mapping_from_dir(train_dir: str, io_txt: str) -> Tuple[dict, list]:
    ordered_names = []
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"not find: {train_dir}")
        
    for letter in sorted(os.listdir(train_dir)):
        letter_dir = os.path.join(train_dir, letter)
        if os.path.isdir(letter_dir):
            for class_name in sorted(os.listdir(letter_dir)):
                if os.path.isdir(os.path.join(letter_dir, class_name)):
                    ordered_names.append(f"/{letter}/{class_name}")

    ordered_names = sorted(ordered_names) 

    io_mapping = {}
    if os.path.exists(io_txt):
        with open(io_txt, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    io_mapping[parts[0]] = max(0, int(parts[1]) - 1)
    else:
        print(f"[!] not find IO_places365.txt")

    mapping_dict = {}
    for fine_id, class_name in enumerate(ordered_names):
        clean_name = class_name.replace('-', '').replace('_', '').replace('/', '')
        matched_c_id = 0
        for io_key, io_val in io_mapping.items():
            clean_io = io_key.replace('-', '').replace('_', '').replace('/', '')
            if clean_name in clean_io or clean_io in clean_name:
                matched_c_id = io_val
                break
        
        mapping_dict[class_name] = (fine_id, matched_c_id)

    print(f"[*] successful!  {len(ordered_names)} ")
    return mapping_dict, ordered_names


class MiniPlacesDataset(Dataset):
    def __init__(self, 
                 root: str,
                 split: str,          # 'train' 或 'val'
                 mapping_dict: dict,  
                 ordered_names: list,
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
        
        self.mean, self.std = mean, std
        self.n_segments, self.compactness = n_segments, compactness
        self.blur_ops, self.scale_factor = blur_ops, scale_factor
        self.is_hier, self.category, self.transform = is_hier, category, transform

        self.img_path = []
        self.fine_labels = []   # L1: 0-99
        self.coarse_labels = [] # L2: 0-1 (Indoor/Outdoor)
        train_dir = os.path.join(root, 'images', 'train')
    
        rng = np.random.RandomState(42) 

        for fine_id, class_name in enumerate(ordered_names):
            coarse_id = mapping_dict[class_name][1]
            class_dir = os.path.join(train_dir, class_name.strip('/'))
            
            if os.path.exists(class_dir):
                current_class_imgs = []
                for img_name in os.listdir(class_dir):
                    if img_name.endswith('.jpg'):
                        current_class_imgs.append(os.path.join(class_dir, img_name))
                
                current_class_imgs = sorted(current_class_imgs)
                rng.shuffle(current_class_imgs)
                split_idx = int(len(current_class_imgs) * split_ratio)
                if split == 'train':
                    selected_imgs = current_class_imgs[:split_idx]
                else:
                    selected_imgs = current_class_imgs[split_idx:]
                for img_path in selected_imgs:
                    self.img_path.append(img_path)
                    self.fine_labels.append(fine_id)
                    self.coarse_labels.append(coarse_id)

        self.targets = self.fine_labels
        print(f"[*] successful! MiniPlaces {split} , A total of {len(self.img_path)} images.")

    def __len__(self):
        return len(self.fine_labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)

        compactness, blur_ops = self.compactness, self.blur_ops
        n_segments, scale_factor = self.n_segments, self.scale_factor
        
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
                    samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2, histogram_bins=5, double_step=False)
                seeds.iterate(samp, num_iterations=15)
                segments.append(torch.LongTensor(seeds.getLabels()))
        else:
            if blur_ops is not None: sample = blur_ops(sample)
            samp = (sample.data.numpy().transpose(1, 2, 0) * self.std + self.mean)
            samp = (samp * 255).astype(np.uint8)
            samp = cv2.cvtColor(samp, cv2.COLOR_RGB2LAB)
            seeds = cv2.ximgproc.createSuperpixelSEEDS(
                samp.shape[1], samp.shape[0], 3, num_superpixels=self.n_segments, num_levels=1, prior=2, histogram_bins=5, double_step=False)
            seeds.iterate(samp, num_iterations=15)
            segments = torch.LongTensor(seeds.getLabels())

        if self.is_hier:
            return sample, segments, self.fine_labels[index], self.coarse_labels[index]
        else:
            if self.category == 'name':
                return sample, segments, self.fine_labels[index]    
            else:
                return sample, segments, self.coarse_labels[index]