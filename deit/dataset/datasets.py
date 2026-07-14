# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.

import os
import csv
import json
import sys

sys.path.append('/data/xiaomeng/model/HCAST-V1/HCAST-V2/deit/dataset')
sys.path.append('/data/xiaomeng/model/HCAST-V1/HCAST-V2/deit')

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

import aircraft_seeds
import aircraft
import dataset.birds_seeds as birds_seeds
import breeds
import breeds_seeds
import Places365
import things
import things_plus
import indoor
import SUN397
import SUN397_2
import MiniPlaces

import clip
from typing import List, Dict, Set, Tuple

INDOOR67_CLASSES = sorted(list(indoor.INDOOR67_SUPER_DICT.keys()))
NAME_TO_FINE_ID = {name: idx for idx, name in enumerate(INDOOR67_CLASSES)}


def get_sun397_mapping(classname_txt: str, excel_file: str) -> dict:
    """
    Returns a mapping { "/a/abbey": (fine_id, mid_id, coarse_id) }.
    """
    mapping_dict = {}
    name_to_fine = {}
    ordered_fine_names = []

    if not os.path.exists(classname_txt):
        print(f"Warning: {classname_txt} not found.")
        return {}

    with open(classname_txt, 'r', encoding='utf-8') as f:
        for fine_id, line in enumerate(f):
            class_name = line.strip()
            name_to_fine[class_name] = fine_id
            ordered_fine_names.append(class_name)

    for name in ordered_fine_names:
        mapping_dict[name] = (name_to_fine[name], 0, 0)

    try:
        df = pd.read_excel(excel_file, header=1)
    except Exception as e:
        print(f"Error reading excel: {e}")
        return mapping_dict

    for index, row in df.iterrows():
        cell_val = str(row.iloc[0]).strip()
        class_name = cell_val.replace("'", "")

        if class_name not in name_to_fine:
            alt_name = class_name[1:] if class_name.startswith('/') else '/' + class_name
            if alt_name in name_to_fine:
                class_name = alt_name
            else:
                continue

        fine_id = name_to_fine[class_name]

        # Coarse ID: 0=indoor, 1=outdoor natural, 2=outdoor man-made
        coarse_id = 0
        if row.iloc[1] == 1:
            coarse_id = 0
        elif row.iloc[2] == 1:
            coarse_id = 1
        elif row.iloc[3] == 1:
            coarse_id = 2

        # Mid ID: columns 4 to 20
        mid_id = 0
        for i in range(4, 21):
            if i < len(row) and row.iloc[i] == 1:
                mid_id = i - 4
                break

        mapping_dict[class_name] = (fine_id, mid_id, coarse_id)

    print(f"[*] Successfully mapped {len(mapping_dict)} categories from SUN397 excel.")
    return mapping_dict


def get_sun397_2_mapping(classname_txt: str, excel_file: str) -> dict:
    """
    Returns a mapping { "/a/abbey": (mid_id, coarse_id) }.
    """
    mapping_dict = {}
    name_to_valid = set()

    if not os.path.exists(classname_txt):
        print(f"Warning: {classname_txt} not found.")
        return {}

    with open(classname_txt, 'r', encoding='utf-8') as f:
        for line in f:
            name_to_valid.add(line.strip())

    for name in name_to_valid:
        mapping_dict[name] = (0, 0)

    try:
        df = pd.read_excel(excel_file, header=1)
    except Exception as e:
        print(f"Error reading excel: {e}")
        return mapping_dict

    for index, row in df.iterrows():
        cell_val = str(row.iloc[0]).strip()
        class_name = cell_val.replace("'", "")

        if class_name not in name_to_valid:
            alt_name = class_name[1:] if class_name.startswith('/') else '/' + class_name
            if alt_name in name_to_valid:
                class_name = alt_name
            else:
                continue

        # Coarse ID: 0=indoor, 1=outdoor natural, 2=outdoor man-made
        coarse_id = 0
        if row.iloc[1] == 1:
            coarse_id = 0
        elif row.iloc[2] == 1:
            coarse_id = 1
        elif row.iloc[3] == 1:
            coarse_id = 2

        # Mid ID: columns 4 to 20
        mid_id = 0
        for i in range(4, 21):
            if i < len(row) and row.iloc[i] == 1:
                mid_id = i - 4
                break

        mapping_dict[class_name] = (mid_id, coarse_id)

    return mapping_dict


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'AIR-HIER':
        dataset = aircraft.FGVCAircraft_Hier(
            args.data_path,
            is_train=is_train,
            transform=transform,
        )
        nb_classes = [100, 70, 30]

    elif args.data_set == 'AIR-HIER-SUPERPIXEL':
        dataset = aircraft_seeds.FGVCAircraft(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [100, 70, 30]

    elif args.data_set == 'BIRD-HIER':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
            random_seed=args.random_seed,
            train=is_train,
        )
        nb_classes = [200, 38, 13]

    elif args.data_set == 'BIRD-HIER-SUPERPIXEL':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = birds_seeds.ImageFolder(
            root,
            transform=transform,
            is_hier=True,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [200, 38, 13]

    elif args.data_set == 'INAT18-HIER-SUPERPIXEL':
        dataset = inat18_seeds.iNatHierDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [8142, 274, 14]

    elif args.data_set == 'INAT21-MINI-HIER':
        dataset = inat21_mini.iNat21MiniDataset(
            args.data_path,
            transform=transform,
            is_hier=True,
            is_train=is_train,
        )
        nb_classes = [10000, 1103, 273]

    elif args.data_set == 'INAT21-MINI-HIER-SUPERPIXEL':
        dataset = inat21_mini_seeds.iNat21MiniDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            mean=[0.466, 0.471, 0.380],
            std=[0.195, 0.194, 0.192],
            n_segments=args.num_superpixels,
            compactness=10.0,
            blur_ops=None,
            scale_factor=1.0,
        )
        nb_classes = [10000, 1103, 273]

    elif args.data_set == 'BREEDS-HIER-SUPERPIXEL':
        dataset = breeds_seeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort=args.breeds_sort,
            path_yn=args.path_yn,
        )
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]

    elif args.data_set == 'BREEDS-HIER':
        dataset = breeds.BreedsDataset(
            args.data_path,
            is_train=is_train,
            transform=transform,
            is_hier=True,
            sort=args.breeds_sort,
            path_yn=args.path_yn,
        )
        if args.breeds_sort == 'entity13':
            nb_classes = [130, 13]
        elif args.breeds_sort == 'living17':
            nb_classes = [34, 17]
        elif args.breeds_sort == 'nonliving26':
            nb_classes = [52, 26]
        elif args.breeds_sort == 'entity30':
            nb_classes = [120, 30]

    elif args.data_set == 'places365':
        data_root = '/data/xiaomeng/dataset/places365/datasets/benjaminkz/places365/versions/1/'
        io_file = os.path.join(data_root, 'IO_places365.txt')
        cat_file = os.path.join(data_root, 'categories_places365.txt')
        if is_train:
            dataset = Places365.Places365Dataset(
                root=data_root,
                txt_file=args.train_txt,
                io_file=io_file,
                cat_file=cat_file,
                transform=transform,
                is_hier=True,
            )
        else:
            dataset = Places365.Places365Dataset(
                root=data_root,
                txt_file=args.val_txt,
                io_file=io_file,
                cat_file=cat_file,
                transform=transform,
                is_hier=True,
            )
        nb_classes = [365, 2]

    elif args.data_set == 'miniplaces':
        data_root = '/data/xiaomeng/dataset/miniplaces/'
        io_txt = '/data/xiaomeng/dataset/places365/datasets/benjaminkz/places365/versions/1/IO_places365.txt'
        train_dir = os.path.join(data_root, 'images', 'train')
        mapping_dict, ordered_names = MiniPlaces.get_miniplaces_mapping_from_dir(train_dir, io_txt)
        nb_classes = [100, 2]

        if is_train:
            dataset = MiniPlaces.MiniPlacesDataset(
                root=data_root,
                split='train',
                mapping_dict=mapping_dict,
                ordered_names=ordered_names,
                split_ratio=0.8,
                transform=transform,
            )
        else:
            dataset = MiniPlaces.MiniPlacesDataset(
                root=data_root,
                split='val',
                mapping_dict=mapping_dict,
                ordered_names=ordered_names,
                split_ratio=0.8,
                transform=transform,
            )

    elif args.data_set == 'things':
        data_root = '/data/xiaomeng/dataset/things/'
        image_paths_file = os.path.join(data_root, '01_image-level/image-paths.csv')
        concepts_file = os.path.join(data_root, '02_object-level/_concepts-metadata_things.tsv')
        category_file = os.path.join(data_root, '03_category-level/category27_manual.tsv')

        dataset = things.ThingsDataset(
            root=data_root,
            image_paths_file=image_paths_file,
            concepts_file=concepts_file,
            category_file=category_file,
            is_train=is_train,
            transform=transform,
        )
        nb_classes = [1854, 27]

    elif args.data_set == 'things_plus':
        data_root = '/data/xiaomeng/dataset/things/'
        image_paths_file = os.path.join(data_root, '01_image-level/image-paths.csv')
        concepts_file = os.path.join(data_root, '02_object-level/_concepts-metadata_things.tsv')
        category_file = os.path.join(data_root, '03_category-level/category53_wide-format.tsv')

        dataset = things_plus.ThingsDataset(
            root=data_root,
            image_paths_file=image_paths_file,
            concepts_file=concepts_file,
            category_file=category_file,
            is_train=is_train,
            transform=transform,
        )
        nb_classes = [1854, 53]

    elif args.data_set == 'indoor':
        data_root = '/data/xiaomeng/dataset/indoorCVPR/'
        train_txt = os.path.join(data_root, 'TrainImages.txt')
        test_txt = os.path.join(data_root, 'TestImages.txt')

        if is_train:
            dataset = indoor.Indoor67Dataset(
                root=data_root,
                txt_file=train_txt,
                is_train=True,
                transform=transform,
            )
        else:
            dataset = indoor.Indoor67Dataset(
                root=data_root,
                txt_file=test_txt,
                is_train=False,
                transform=transform,
            )
        nb_classes = [67, 5]

    elif args.data_set == 'sun397':
        data_root = '/data/xiaomeng/dataset/SUN397/'
        train_txt = os.path.join(data_root, 'Partitions', 'Training_01.txt')
        test_txt = os.path.join(data_root, 'Partitions', 'Testing_01.txt')
        classname_txt = os.path.join(data_root, 'Partitions', 'ClassName.txt')
        excel_file = os.path.join(data_root, 'hierarchy_three_levels', 'three_levels.xlsx')

        mapping_dict = get_sun397_mapping(classname_txt, excel_file)
        if is_train:
            dataset = SUN397.SUN397Dataset(
                root=data_root,
                txt_file=train_txt,
                mapping_dict=mapping_dict,
                is_train=is_train,
                transform=transform,
            )
        else:
            dataset = SUN397.SUN397Dataset(
                root=data_root,
                txt_file=test_txt,
                mapping_dict=mapping_dict,
                is_train=is_train,
                transform=transform,
            )
        nb_classes = [397, 16, 3]

    elif args.data_set == 'sun397_2':
        data_root = '/data/xiaomeng/dataset/SUN397/'
        train_txt = os.path.join(data_root, 'Partitions', 'Training_01.txt')
        test_txt = os.path.join(data_root, 'Partitions', 'Testing_01.txt')
        classname_txt = os.path.join(data_root, 'Partitions', 'ClassName.txt')
        excel_file = os.path.join(data_root, 'hierarchy_three_levels', 'three_levels.xlsx')
        mapping_dict = get_sun397_2_mapping(classname_txt, excel_file)
        nb_classes = [16, 3]
        txt_file = train_txt if is_train else test_txt
        dataset = SUN397_2.SUN397_2Dataset(
            root=data_root,
            txt_file=txt_file,
            mapping_dict=mapping_dict,
            is_train=is_train,
            transform=transform,
            is_hier=True,
        )

    return dataset, nb_classes


def build_hierarchy_map_entity13(class_labels: List[int], super_labels: List[int]) -> Dict[int, List[int]]:
    """
    Key: Coarse ID
    Value: List of Fine IDs
    """
    hierarchy_map: Dict[int, Set[int]] = {}
    for fine_id, coarse_id in zip(class_labels, super_labels):
        if coarse_id not in hierarchy_map:
            hierarchy_map[coarse_id] = set()
        hierarchy_map[coarse_id].add(fine_id)
    final_map = {c_id: sorted(list(f_ids)) for c_id, f_ids in hierarchy_map.items()}
    return final_map


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if 'INAT' in args.data_set:
        t.append(transforms.Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192]))
    else:
        t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)


def build_hierarchy_map(class_labels: List[int], super_labels: List[int]) -> Dict[int, List[int]]:
    """
    Args:
        class_labels: Fine IDs
        super_labels: Coarse IDs
    Returns:
        Dict: {coarse_id: [fine_id1, fine_id2, ...]}
    """
    hierarchy_map_dict: Dict[int, Set[int]] = {}
    for fine_id, coarse_id in zip(class_labels, super_labels):
        if coarse_id not in hierarchy_map_dict:
            hierarchy_map_dict[coarse_id] = set()
        hierarchy_map_dict[coarse_id].add(fine_id)
    final_map = {c_id: sorted(list(f_ids)) for c_id, f_ids in hierarchy_map_dict.items()}
    return final_map


def hierarchy_map(args):
    final_hierarchy_maps = []
    if args.data_set == 'BREEDS-HIER-SUPERPIXEL':
        if args.breeds_sort == 'entity13':
            fine_labels = list(range(130))
            coarse_labels = []
            for coarse_id in range(13):
                coarse_labels.extend([coarse_id] * 10)

            h_map = build_hierarchy_map(fine_labels, coarse_labels)
            final_hierarchy_maps = [h_map]

        elif args.breeds_sort == 'entity30':
            fine_labels = list(range(120))
            coarse_labels = []
            for coarse_id in range(30):
                coarse_labels.extend([coarse_id] * 4)

            h_map = build_hierarchy_map(fine_labels, coarse_labels)
            final_hierarchy_maps = [h_map]

        elif args.breeds_sort == 'living17':
            fine_labels = list(range(34))
            coarse_labels = []
            for coarse_id in range(17):
                coarse_labels.extend([coarse_id] * 2)

            h_map = build_hierarchy_map(fine_labels, coarse_labels)
            final_hierarchy_maps = [h_map]

        elif args.breeds_sort == 'nonliving26':
            fine_labels = list(range(52))
            coarse_labels = []
            for coarse_id in range(26):
                coarse_labels.extend([coarse_id] * 2)

            h_map = build_hierarchy_map(fine_labels, coarse_labels)
            final_hierarchy_maps = [h_map]

    elif args.data_set == 'places365':
        data_root = '/data/xiaomeng/dataset/places365/datasets/benjaminkz/places365/versions/1/'
        io_file = os.path.join(data_root, 'IO_places365.txt')
        cat_file = os.path.join(data_root, 'categories_places365.txt')

        name_to_fine = {}
        with open(cat_file, 'r') as f:
            for line in f:
                name, fine_id = line.strip().split()
                name_to_fine[name] = int(fine_id)

        fine_labels = []
        coarse_labels = []
        with open(io_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                name = parts[0]
                coarse_id = int(parts[1]) - 1  # map 1,2 to 0,1

                if name in name_to_fine:
                    fine_labels.append(name_to_fine[name])
                    coarse_labels.append(coarse_id)

        h_map = build_hierarchy_map(fine_labels, coarse_labels)
        final_hierarchy_maps = [h_map]

    elif args.data_set == 'things':
        # {coarse_id: [fine_id1, fine_id2, ...]}
        hierarchy_dict = {i: [] for i in range(27)}

        with open('/data/xiaomeng/dataset/things/03_category-level/category27_manual.tsv', 'r', encoding='utf-8') as f:
            _ = f.readline()
            fine_id = 0
            for line in f:
                parts = line.strip().split('\t')
                try:
                    coarse_id = parts.index('1')
                    hierarchy_dict[coarse_id].append(fine_id)
                except ValueError:
                    pass
                fine_id += 1
        final_hierarchy_maps = [hierarchy_dict]

    elif args.data_set == 'things_plus':
        # {coarse_id: [fine_id1, fine_id2, ...]}
        hierarchy_dict = {i: [] for i in range(53)}
        tsv_path = '/data/xiaomeng/dataset/things/03_category-level/category53_wide-format.tsv'
        with open(tsv_path, 'r', encoding='utf-8') as f:
            _ = f.readline()
            fine_id = 0
            for line in f:
                parts = line.strip().split('\t')
                coarse_parts = parts[2:]
                found = False
                for coarse_id, val in enumerate(coarse_parts):
                    if val == '1' and coarse_id < 53:
                        hierarchy_dict[coarse_id].append(fine_id)
                        found = True
                if not found:
                    hierarchy_dict[0].append(fine_id)
                fine_id += 1
        for i in range(53):
            if len(hierarchy_dict[i]) == 0:
                hierarchy_dict[i].append(0)
        final_hierarchy_maps = [hierarchy_dict]

    elif args.data_set == 'indoor':
        hierarchy_dict = {i: [] for i in range(5)}
        for fine_id, class_name in enumerate(INDOOR67_CLASSES):
            coarse_id = indoor.INDOOR67_SUPER_DICT[class_name]
            hierarchy_dict[coarse_id].append(fine_id)
        final_hierarchy_maps = [hierarchy_dict]

    return final_hierarchy_maps


# ========================= CLIP ====================================
CLIP_EMBED_DIM = 512  # ViT-B/16 embedding dimension

ENTITY13_LABELS_MAP = {
    "0": [869, "n04479046", "trench coat"], "1": [399, "n02667093", "abaya"],
    "2": [578, "n03450230", "gown"], "3": [735, "n03980874", "poncho"],
    "4": [652, "n03763968", "military uniform"], "5": [610, "n03595614", "jersey, T-shirt, tee shirt"],
    "6": [501, "n03045698", "cloak"], "7": [445, "n02837789", "bikini, two-piece"],
    "8": [655, "n03770439", "miniskirt, mini"], "9": [842, "n04371430", "swimming trunks, bathing trunks"],
    "10": [87, "n01817953", "African grey, African gray, Psittacus erithacus"],
    "11": [92, "n01828970", "bee eater"], "12": [91, "n01824575", "coucal"],
    "13": [137, "n02018207", "American coot, marsh hen, mud hen, water hen, Fulica americana"],
    "14": [14, "n01537544", "indigo bunting, indigo finch, indigo bird, Passerina cyanea"],
    "15": [145, "n02056570", "king penguin, Aptenodytes patagonica"],
    "16": [129, "n02006656", "spoonbill"], "17": [135, "n02013706", "limpkin, Aramus pictus"],
    "18": [85, "n01806567", "quail"], "19": [21, "n01608432", "kite"],
    "20": [45, "n01692333", "Gila monster, Heloderma suspectum"],
    "21": [42, "n01687978", "agama"], "22": [51, "n01704323", "triceratops"],
    "23": [47, "n01694178", "African chameleon, Chamaeleo chamaeleon"],
    "24": [52, "n01728572", "thunder snake, worm snake, Carphophis amoenus"],
    "25": [63, "n01748264", "Indian cobra, Naja naja"],
    "26": [55, "n01729977", "green snake, grass snake"],
    "27": [35, "n01667114", "mud turtle"],
    "28": [58, "n01737021", "water snake"],
    "29": [33, "n01664065", "loggerhead, loggerhead turtle, Caretta caretta"],
    "30": [119, "n01978455", "rock crab, Cancer irroratus"],
    "31": [72, "n01773157", "black and gold garden spider, Argiope aurantia"],
    "32": [300, "n02165105", "tiger beetle"],
    "33": [75, "n01774384", "black widow, Latrodectus mactans"],
    "34": [73, "n01773549", "barn spider, Araneus cavaticus"],
    "35": [317, "n02259212", "leafhopper"],
    "36": [302, "n02167151", "ground beetle, carabid beetle"],
    "37": [120, "n01980166", "fiddler crab"],
    "38": [309, "n02206856", "bee"],
    "39": [313, "n02231487", "walking stick, walkingstick, stick insect"],
    "40": [284, "n02123597", "Siamese cat, Siamese"],
    "41": [350, "n02417914", "ibex, Capra ibex"],
    "42": [292, "n02129604", "tiger, Panthera tigris"],
    "43": [344, "n02398521", "hippopotamus, hippo, river horse, Hippopotamus amphibius"],
    "44": [174, "n02091467", "Norwegian elkhound, elkhound"],
    "45": [149, "n02074367", "dugong, Dugong dugon"],
    "46": [375, "n02488702", "colobus, colobus monkey"],
    "47": [258, "n02111889", "Samoyed, Samoyede"],
    "48": [283, "n02123394", "Persian cat"],
    "49": [170, "n02090721", "Irish wolfhound"],
    "50": [443, "n02834397", "bib"],
    "51": [552, "n03325584", "feather boa, boa"],
    "52": [824, "n04325704", "stole"],
    "53": [728, "n03958227", "plastic bag"],
    "54": [433, "n02807133", "bathing cap, swimming cap"],
    "55": [514, "n03124043", "cowboy boot"],
    "56": [679, "n03814906", "necklace"],
    "57": [518, "n03127747", "crash helmet"],
    "58": [570, "n03424325", "gasmask, respirator, gas helmet"],
    "59": [638, "n03710637", "maillot"],
    "60": [484, "n02981792", "catamaran"],
    "61": [814, "n04273569", "speedboat"],
    "62": [554, "n03344393", "fireboat"],
    "63": [914, "n04612504", "yawl"],
    "64": [404, "n02690373", "airliner"],
    "65": [510, "n03095699", "container ship, containership, container vessel"],
    "66": [628, "n03673027", "liner, ocean liner"],
    "67": [871, "n04483307", "trimaran"],
    "68": [812, "n04266014", "space shuttle"],
    "69": [403, "n02687172", "aircraft carrier, carrier, flattop, attack aircraft carrier"],
    "70": [890, "n04540053", "volleyball"],
    "71": [681, "n03832673", "notebook, notebook computer"],
    "72": [430, "n02802426", "basketball"],
    "73": [590, "n03485407", "hand-held computer, hand-held microcomputer"],
    "74": [872, "n04485082", "tripod"],
    "75": [745, "n04009552", "projector"],
    "76": [422, "n02790996", "barbell"],
    "77": [664, "n03782006", "monitor"],
    "78": [522, "n03134739", "croquet ball"],
    "79": [416, "n02777292", "balance beam, beam"],
    "80": [894, "n04550184", "wardrobe, closet, press"],
    "81": [861, "n04447861", "toilet seat"],
    "82": [553, "n03337140", "file, file cabinet, filing cabinet"],
    "83": [669, "n03788365", "mosquito net"],
    "84": [564, "n03388549", "four-poster"],
    "85": [431, "n02804414", "bassinet"],
    "86": [493, "n03016953", "chiffonier, commode"],
    "87": [559, "n03376595", "folding chair"],
    "88": [556, "n03347037", "fire screen, fireguard"],
    "89": [789, "n04201297", "shoji"],
    "90": [881, "n04515003", "upright, upright piano"],
    "91": [695, "n03874599", "padlock"],
    "92": [626, "n03666591", "lighter, light, igniter, ignitor"],
    "93": [822, "n04311174", "steel drum"],
    "94": [704, "n03891332", "parking meter"],
    "95": [499, "n03041632", "cleaver, meat cleaver, chopper"],
    "96": [845, "n04376876", "syringe"],
    "97": [398, "n02666196", "abacus"],
    "98": [778, "n04141975", "scale, weighing machine"],
    "99": [512, "n03109150", "corkscrew, bottle screw"],
    "100": [483, "n02980441", "castle"],
    "101": [442, "n02825657", "bell cote, bell cot"],
    "102": [562, "n03388043", "fountain"],
    "103": [727, "n03956157", "planetarium"],
    "104": [920, "n06874185", "traffic light, traffic signal, stoplight"],
    "105": [460, "n02894605", "breakwater, groin, groyne, mole, bulwark, seawall, jetty"],
    "106": [500, "n03042490", "cliff dwelling"],
    "107": [663, "n03781244", "monastery"],
    "108": [743, "n04005630", "prison, prison house"],
    "109": [900, "n04562935", "water tower"],
    "110": [803, "n04252225", "snowplow, snowplough"],
    "111": [867, "n04467665", "trailer truck, tractor trailer, trucking rig, rig, articulated lorry, semi"],
    "112": [751, "n04037443", "racer, race car, racing car"],
    "113": [791, "n04204347", "shopping cart"],
    "114": [880, "n04509417", "unicycle, monocycle"],
    "115": [670, "n03791053", "motor scooter, scooter"],
    "116": [705, "n03895866", "passenger car, coach, carriage"],
    "117": [654, "n03769881", "minibus"],
    "118": [609, "n03594945", "jeep, landrover"],
    "119": [757, "n04065272", "recreational vehicle, RV, R.V."],
    "120": [937, "n07714990", "broccoli"],
    "121": [987, "n12144580", "corn"],
    "122": [950, "n07747607", "orange"],
    "123": [943, "n07718472", "cucumber, cuke"],
    "124": [940, "n07716906", "spaghetti squash"],
    "125": [942, "n07717556", "butternut squash"],
    "126": [941, "n07717410", "acorn squash"],
    "127": [938, "n07715103", "cauliflower"],
    "128": [945, "n07720875", "bell pepper"],
    "129": [952, "n07753113", "fig"]
}

COARSE_LABELS_MAP = {
    "0": "garment", "1": "bird", "2": "reptile, reptilian", "3": "arthropod",
    "4": "mammal, mammalian", "5": "accessory, accoutrement, accouterment",
    "6": "craft", "7": "equipment", "8": "furniture, piece of furniture, article of furniture",
    "9": "instrument", "10": "man-made structure, construction", "11": "wheeled vehicle",
    "12": "produce, green goods, green groceries, garden truck"
}

CLIP_MODEL_PATH = "/data/xiaomeng/model/HCAST-main/deit/ViT-B-16.pt"


def get_entity13_fine_class_names(num_fine_classes: int) -> List[str]:
    class_names: List[str] = []
    for i in range(num_fine_classes):
        label_info = ENTITY13_LABELS_MAP[str(i)]
        raw_label = label_info[2]
        templated_label = f"a photo of a {raw_label}"
        class_names.append(templated_label)
    return class_names


def get_coarse_class_names(coarse_labels_map: dict = COARSE_LABELS_MAP) -> List[str]:
    class_names: List[str] = []
    sorted_keys = sorted(coarse_labels_map.keys(), key=lambda x: int(x))
    for key in sorted_keys:
        raw_label = coarse_labels_map[key]
        templated_label = f"a photo of a {raw_label}"
        class_names.append(templated_label)
    return class_names


def load_clip_and_encode_text(class_names: List[str], device: str = "cpu") -> torch.Tensor:
    print(f"--- Loading CLIP and encoding text (embed dim: {CLIP_EMBED_DIM}) ---")

    model, preprocess = clip.load("ViT-B/16", device=device, jit=False)

    if os.path.exists(CLIP_MODEL_PATH):
        print(f"Attempting to load weights from: {CLIP_MODEL_PATH}")
        checkpoint = torch.load(CLIP_MODEL_PATH, map_location=device)

        # Determine the format of the checkpoint
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        elif isinstance(checkpoint, (torch.jit.ScriptModule, torch.jit.RecursiveScriptModule)):
            print("Warning: extracting state_dict from JIT ScriptModule.")
            state_dict = checkpoint.state_dict()
        else:
            print(f"Error: unrecognized checkpoint format in {CLIP_MODEL_PATH}.")
            state_dict = None

        if state_dict is not None:
            new_state_dict = {}
            # Remove non-weight keys that cause loading errors
            keys_to_remove = ["input_resolution", "context_length", "vocab_size"]
            for k, v in state_dict.items():
                if k in keys_to_remove:
                    print(f"    - Removing config key: {k}")
                    continue
                if k.startswith('module.'):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            try:
                model.load_state_dict(new_state_dict, strict=True)
                print("Local weights loaded successfully (strict=True).")
            except RuntimeError as e:
                if "size mismatch" in str(e):
                    print("Warning: size mismatch, falling back to strict=False.")
                    model.load_state_dict(new_state_dict, strict=False)
                    print("Local weights loaded successfully (strict=False).")
                else:
                    raise e
        else:
            print(f"Warning: local weights not found at {CLIP_MODEL_PATH}. Using default weights.")
    # Encode text
    text_inputs = clip.tokenize(class_names).to(device)
    with torch.no_grad():
        clip_embeds = model.encode_text(text_inputs).float()
    clip_embeds = F.normalize(clip_embeds, p=2, dim=1)

    print(f"Generated CLIP embeddings, shape: {clip_embeds.shape}")
    return clip_embeds


def Prototype_initialization():
    NUM_FINE_CLASSES = 130  # Entity13 L1 class count
    fine_class_names = get_entity13_fine_class_names(NUM_FINE_CLASSES)
    coarse_class_names = get_coarse_class_names()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    fine_clip_prototypes = load_clip_and_encode_text(fine_class_names, device)
    coarse_clip_prototypes = load_clip_and_encode_text(coarse_class_names, device)

    MODEL_EMBED_DIM = 384

    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False)
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
            coarse_prototypes_projected = projection_layer(coarse_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
        coarse_clip_embeds = coarse_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes
        coarse_clip_embeds = coarse_clip_prototypes

    return fine_clip_embeds, coarse_clip_embeds


def get_cub_fine_grained_class_names(
    file_path: str = "/data/xiaomeng/dataset/cross_domian/CUB_200_2011/classes.txt"
) -> List[str]:
    """
    Read fine-grained class names from CUB-200-2011 classes.txt and apply CLIP prompt template.
    """
    class_names: List[str] = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    raw_label_with_prefix = parts[1]
                    if '.' in raw_label_with_prefix:
                        raw_label = raw_label_with_prefix.split('.', 1)[1].replace('_', ' ')
                    else:
                        raw_label = raw_label_with_prefix.replace('_', ' ')
                    templated_label = f"a photo of a {raw_label}"
                    class_names.append(templated_label)
    except FileNotFoundError:
        print(f"Error: file not found at {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    return class_names


def Prototype_initialization_CUB():
    fine_class_names = get_cub_fine_grained_class_names()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fine_clip_prototypes = load_clip_and_encode_text(fine_class_names, device)

    MODEL_EMBED_DIM = 384
    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False)
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes

    return fine_clip_embeds


# ========================= Entity-30 ====================================
ENTITY30_LABELS_MAP = {
    "0": [64, "n01749939", "green mamba"], "1": [56, "n01734418", "king snake, kingsnake"],
    "2": [57, "n01735189", "garter snake, grass snake"],
    "3": [52, "n01728572", "thunder snake, worm snake, Carphophis amoenus"],
    "4": [11, "n01531178", "goldfinch, Carduelis carduelis"],
    "5": [10, "n01530575", "brambling, Fringilla montifringilla"],
    "6": [20, "n01601694", "water ouzel, dipper"], "7": [19, "n01592084", "chickadee"],
    "8": [44, "n01689811", "alligator lizard"],
    "9": [45, "n01692333", "Gila monster, Heloderma suspectum"],
    "10": [40, "n01682714", "American chameleon, anole, Anolis carolinensis"],
    "11": [46, "n01693334", "green lizard, Lacerta viridis"],
    "12": [70, "n01770081", "harvestman, daddy longlegs, Phalangium opilio"],
    "13": [73, "n01773549", "barn spider, Araneus cavaticus"],
    "14": [71, "n01770393", "scorpion"],
    "15": [75, "n01774384", "black widow, Latrodectus mactans"],
    "16": [146, "n02058221", "albatross, mollymawk"],
    "17": [140, "n02027492", "red-backed sandpiper, dunlin, Erolia alpina"],
    "18": [134, "n02012849", "crane"],
    "19": [127, "n02002556", "white stork, Ciconia ciconia"],
    "20": [124, "n01985128", "crayfish, crawfish, crawdad, crawdaddy"],
    "21": [123, "n01984695", "spiny lobster, langouste, rock lobster, crawfish, crayfish, sea crawfish"],
    "22": [125, "n01986214", "hermit crab"],
    "23": [118, "n01978287", "Dungeness crab, Cancer magister"],
    "24": [171, "n02091032", "Italian greyhound"],
    "25": [359, "n02443484", "black-footed ferret, ferret, Mustela nigripes"],
    "26": [181, "n02093647", "Bedlington terrier"],
    "27": [253, "n02110806", "basenji"],
    "28": [318, "n02264363", "lacewing, lacewing fly"],
    "29": [308, "n02190166", "fly"],
    "30": [311, "n02226429", "grasshopper, hopper"],
    "31": [325, "n02281406", "sulphur butterfly, sulfur butterfly"],
    "32": [355, "n02437616", "llama"],
    "33": [353, "n02423022", "gazelle"],
    "34": [340, "n02391049", "zebra"],
    "35": [345, "n02403003", "ox"],
    "36": [372, "n02486410", "baboon"],
    "37": [379, "n02492660", "howler monkey, howler"],
    "38": [383, "n02497673", "Madagascar cat, ring-tailed lemur, Lemur catta"],
    "39": [367, "n02481823", "chimpanzee, chimp, Pan troglodytes"],
    "40": [391, "n02536864", "coho, cohoe, coho salmon, blue jack, silver salmon, Oncorhynchus kisutch"],
    "41": [0, "n01440764", "tench, Tinca tinca"],
    "42": [396, "n02643566", "lionfish"],
    "43": [392, "n02606052", "rock beauty, Holocanthus tricolor"],
    "44": [460, "n02894605", "breakwater, groin, groyne, mole, bulwark, seawall, jetty"],
    "45": [716, "n03930313", "picket fence, paling"],
    "46": [877, "n04501370", "turnstile"],
    "47": [421, "n02788148", "bannister, banister, balustrade, balusters, handrail"],
    "48": [454, "n02871525", "bookshop, bookstore, bookstall"],
    "49": [483, "n02980441", "castle"],
    "50": [668, "n03788195", "mosque"],
    "51": [467, "n02927161", "butcher shop, meat market"],
    "52": [742, "n04004767", "printer"],
    "53": [707, "n03902125", "pay-phone, pay-station"],
    "54": [650, "n03759954", "microphone, mike"],
    "55": [508, "n03085013", "computer keyboard, keypad"],
    "56": [502, "n03047690", "clog, geta, patten, sabot"],
    "57": [630, "n03680355", "Loafer"],
    "58": [638, "n03710637", "maillot"],
    "59": [770, "n04120489", "running shoe"],
    "60": [400, "n02669723", "academic gown, academic robe, judge's robe"],
    "61": [411, "n02730930", "apron"],
    "62": [655, "n03770439", "miniskirt, mini"],
    "63": [568, "n03404251", "fur coat"],
    "64": [715, "n03929855", "pickelhaube"],
    "65": [584, "n03476684", "hair slide"],
    "66": [793, "n04209133", "shower cap"],
    "67": [452, "n02869837", "bonnet, poke bonnet"],
    "68": [897, "n04554684", "washer, automatic washer, washing machine"],
    "69": [651, "n03761084", "microwave, microwave oven"],
    "70": [521, "n03133878", "Crock Pot"],
    "71": [882, "n04517823", "vacuum, vacuum cleaner"],
    "72": [647, "n03733805", "measuring cup"],
    "73": [499, "n03041632", "cleaver, meat cleaver, chopper"],
    "74": [505, "n03063689", "coffeepot"],
    "75": [813, "n04270147", "spatula"],
    "76": [531, "n03197337", "digital watch"],
    "77": [409, "n02708093", "analog clock"],
    "78": [704, "n03891332", "parking meter"],
    "79": [635, "n03706229", "magnetic compass"],
    "80": [627, "n03670208", "limousine, limo"],
    "81": [779, "n04146614", "school bus"],
    "82": [665, "n03785016", "moped"],
    "83": [511, "n03100240", "convertible"],
    "84": [566, "n03394916", "French horn, horn"],
    "85": [641, "n03720891", "maraca"],
    "86": [579, "n03452741", "grand piano, grand"],
    "87": [881, "n04515003", "upright, upright piano"],
    "88": [552, "n03325584", "feather boa, boa"],
    "89": [678, "n03814639", "neck brace"],
    "90": [443, "n02834397", "bib"],
    "91": [906, "n04591157", "Windsor tie"],
    "92": [795, "n04228054", "ski"],
    "93": [543, "n03255030", "dumbbell"],
    "94": [522, "n03134739", "croquet ball"],
    "95": [752, "n04039381", "racket, racquet"],
    "96": [659, "n03775546", "mixing bowl"],
    "97": [899, "n04560804", "water jug"],
    "98": [441, "n02823750", "beer glass"],
    "99": [898, "n04557648", "water bottle"],
    "100": [749, "n04033901", "quill, quill pen"],
    "101": [507, "n03075370", "combination lock"],
    "102": [695, "n03874599", "padlock"],
    "103": [783, "n04153751", "screw"],
    "104": [510, "n03095699", "container ship, containership, container vessel"],
    "105": [625, "n03662601", "lifeboat"],
    "106": [403, "n02687172", "aircraft carrier, carrier, flattop, attack aircraft carrier"],
    "107": [871, "n04483307", "trimaran"],
    "108": [964, "n07875152", "potpie"],
    "109": [935, "n07711569", "mashed potato"],
    "110": [963, "n07873807", "pizza, pizza pie"],
    "111": [933, "n07697313", "cheeseburger"],
    "112": [939, "n07716358", "zucchini, courgette"],
    "113": [943, "n07718472", "cucumber, cuke"],
    "114": [942, "n07717556", "butternut squash"],
    "115": [944, "n07718747", "artichoke, globe artichoke"],
    "116": [949, "n07745940", "strawberry"],
    "117": [953, "n07753275", "pineapple, ananas"],
    "118": [955, "n07754684", "jackfruit, jak, jack"],
    "119": [948, "n07742313", "Granny Smith"]
}


def get_entity30_fine_class_names(num_fine_classes: int) -> List[str]:
    class_names: List[str] = []
    for i in range(num_fine_classes):
        label_info = ENTITY30_LABELS_MAP[str(i)]
        raw_label = label_info[2]
        templated_label = f"a photo of a {raw_label}"
        class_names.append(templated_label)
    return class_names


def Prototype_initialization_entity30():
    NUM_FINE_CLASSES = 120
    fine_class_names = get_entity30_fine_class_names(NUM_FINE_CLASSES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(fine_class_names, device)

    MODEL_EMBED_DIM = 384
    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False)
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes

    return fine_clip_embeds


# ========================= Living ====================================
living_LABELS_MAP = {
    "0": [27, "n01631663", "eft"], "1": [29, "n01632777", "axolotl, mud puppy, Ambystoma mexicanum"],
    "2": [37, "n01669191", "box turtle, box tortoise"],
    "3": [34, "n01665541", "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea"],
    "4": [41, "n01685808", "whiptail, whiptail lizard"],
    "5": [44, "n01689811", "alligator lizard"],
    "6": [60, "n01740131", "night snake, Hypsiglena torquata"],
    "7": [57, "n01735189", "garter snake, grass snake"],
    "8": [76, "n01774750", "tarantula"],
    "9": [72, "n01773157", "black and gold garden spider, Argiope aurantia"],
    "10": [81, "n01796340", "ptarmigan"],
    "11": [83, "n01798484", "prairie chicken, prairie grouse, prairie fowl"],
    "12": [88, "n01818515", "macaw"],
    "13": [90, "n01820546", "lorikeet"],
    "14": [118, "n01978287", "Dungeness crab, Cancer magister"],
    "15": [120, "n01980166", "fiddler crab"],
    "16": [163, "n02088466", "bloodhound, sleuthhound"],
    "17": [154, "n02086079", "Pekinese, Pekingese, Peke"],
    "18": [272, "n02114855", "coyote, prairie wolf, brush wolf, Canis latrans"],
    "19": [271, "n02114712", "red wolf, maned wolf, Canis rufus, Canis niger"],
    "20": [280, "n02120505", "grey fox, gray fox, Urocyon cinereoargenteus"],
    "21": [279, "n02120079", "Arctic fox, white fox, Alopex lagopus"],
    "22": [282, "n02123159", "tiger cat"],
    "23": [285, "n02124075", "Egyptian cat"],
    "24": [297, "n02134418", "sloth bear, Melursus ursinus, Ursus ursinus"],
    "25": [295, "n02133161", "American black bear, black bear, Ursus americanus, Euarctos americanus"],
    "26": [305, "n02172182", "dung beetle"],
    "27": [306, "n02174001", "rhinoceros beetle"],
    "28": [325, "n02281406", "sulphur butterfly, sulfur butterfly"],
    "29": [321, "n02276258", "admiral"],
    "30": [368, "n02483362", "gibbon, Hylobates lar"],
    "31": [365, "n02480495", "orangutan, orang, orangutang, Pongo pygmaeus"],
    "32": [377, "n02490219", "marmoset"],
    "33": [380, "n02493509", "titi, titi monkey"]
}


def get_living_fine_class_names(num_fine_classes: int) -> List[str]:
    class_names: List[str] = []
    for i in range(num_fine_classes):
        label_info = living_LABELS_MAP[str(i)]
        raw_label = label_info[2]
        templated_label = f"a photo of a {raw_label}"
        class_names.append(templated_label)
    return class_names


def Prototype_initialization_living():
    NUM_FINE_CLASSES = 34
    fine_class_names = get_living_fine_class_names(NUM_FINE_CLASSES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(fine_class_names, device)

    MODEL_EMBED_DIM = 384
    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False)
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes

    return fine_clip_embeds


# ========================= Non-living ====================================
nonliving_LABELS_MAP = {
    "0": [728, "n03958227", "plastic bag"], "1": [748, "n04026417", "purse"],
    "2": [890, "n04540053", "volleyball"],
    "3": [747, "n04023962", "punching bag, punch bag, punching ball, punchball"],
    "4": [576, "n03447447", "gondola"], "5": [871, "n04483307", "trimaran"],
    "6": [465, "n02916936", "bulletproof vest"],
    "7": [461, "n02895154", "breastplate, aegis, egis"],
    "8": [737, "n03983396", "pop bottle, soda bottle"],
    "9": [440, "n02823428", "beer bottle"],
    "10": [874, "n04487081", "trolleybus, trolley coach, trackless trolley"],
    "11": [654, "n03769881", "minibus"],
    "12": [751, "n04037443", "racer, race car, racing car"],
    "13": [661, "n03777568", "Model T"],
    "14": [559, "n03376595", "folding chair"],
    "15": [857, "n04429376", "throne"],
    "16": [617, "n03630383", "lab coat, laboratory coat"],
    "17": [568, "n03404251", "fur coat"],
    "18": [620, "n03642806", "laptop, laptop computer"],
    "19": [527, "n03180011", "desktop computer"],
    "20": [698, "n03877845", "palace"],
    "21": [663, "n03781244", "monastery"],
    "22": [912, "n04604644", "worm fence, snake fence, snake-rail fence, Virginia fence"],
    "23": [489, "n03000134", "chainlink fence"],
    "24": [439, "n02817516", "bearskin, busby, shako"],
    "25": [452, "n02869837", "bonnet, poke bonnet"],
    "26": [579, "n03452741", "grand piano, grand"],
    "27": [687, "n03854065", "organ, pipe organ"],
    "28": [467, "n02927161", "butcher shop, meat market"],
    "29": [424, "n02791270", "barbershop"],
    "30": [580, "n03457902", "greenhouse, nursery, glasshouse"],
    "31": [410, "n02727426", "apiary, bee house"],
    "32": [822, "n04311174", "steel drum"],
    "33": [642, "n03721384", "marimba, xylophone"],
    "34": [849, "n04398044", "teapot"],
    "35": [544, "n03259280", "Dutch oven"],
    "36": [538, "n03220513", "dome"],
    "37": [884, "n04523525", "vault"],
    "38": [780, "n04147183", "schooner"],
    "39": [724, "n03947888", "pirate, pirate ship"],
    "40": [601, "n03534580", "hoopskirt, crinoline"],
    "41": [655, "n03770439", "miniskirt, mini"],
    "42": [546, "n03272010", "electric guitar"],
    "43": [420, "n02787622", "banjo"],
    "44": [531, "n03197337", "digital watch"],
    "45": [826, "n04328186", "stopwatch, stop watch"],
    "46": [555, "n03345487", "fire engine, fire truck"],
    "47": [717, "n03930630", "pickup, pickup truck"],
    "48": [683, "n03838899", "oboe, hautboy, hautbois"],
    "49": [776, "n04141076", "sax, saxophone"],
    "50": [940, "n07716906", "spaghetti squash"],
    "51": [941, "n07717410", "acorn squash"]
}


def get_nonliving_fine_class_names(num_fine_classes: int) -> List[str]:
    class_names: List[str] = []
    for i in range(num_fine_classes):
        label_info = nonliving_LABELS_MAP[str(i)]
        raw_label = label_info[2]
        templated_label = f"a photo of a {raw_label}"
        class_names.append(templated_label)
    return class_names


def Prototype_initialization_nonliving():
    NUM_FINE_CLASSES = 52
    fine_class_names = get_nonliving_fine_class_names(NUM_FINE_CLASSES)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(fine_class_names, device)

    MODEL_EMBED_DIM = 384
    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False)
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes

    return fine_clip_embeds


# ========================= Aircraft ====================================
def get_aircraft_class_names_from_csv(
    file_path: str = "/data/xiaomeng/model/HCAST-V1/HCAST-V2/data/Air.csv"
) -> List[str]:
    """Extract class names from Air.csv."""
    class_names: List[str] = []
    try:
        with open(file_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and len(row) >= 1:
                    raw_label = row[0].strip()
                    if raw_label:
                        class_names.append(raw_label)
    except FileNotFoundError:
        print(f"Error: file not found at {file_path}")
        return []
    except csv.Error as e:
        print(f"CSV error: {e}")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    return class_names


def Prototype_initialization_AIR():
    fine_class_names = get_aircraft_class_names_from_csv()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(fine_class_names, device)

    MODEL_EMBED_DIM = 384
    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False)
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes

    return fine_clip_embeds


def build_hierarchy_map_air(
    lower_labels: List[int], upper_labels: List[int]
) -> Dict[int, List[int]]:
    """
    Build hierarchy mapping from lower level IDs to upper level IDs.
    Key: Upper ID, Value: list of Lower IDs.
    """
    hierarchy_map: Dict[int, Set[int]] = {}
    for lower_id, upper_id in zip(lower_labels, upper_labels):
        if upper_id not in hierarchy_map:
            hierarchy_map[upper_id] = set()
        hierarchy_map[upper_id].add(lower_id)
    final_map = {u_id: sorted(list(l_ids)) for u_id, l_ids in hierarchy_map.items()}
    return final_map


def hierarchy_maps_air(trees: List[List[int]]) -> List[Dict[int, List[int]]]:
    """
    Build multi-level hierarchy maps from the trees list.
    Assumes trees structure: [L1_ID (finest), L2_ID, L3_ID].
    """
    if not trees:
        return []

    # Convert from 1-based to 0-based IDs
    l1_ids: List[int] = [t[0] - 1 for t in trees]
    l2_ids: List[int] = [t[1] - 1 for t in trees]
    l3_ids: List[int] = [t[2] - 1 for t in trees]

    # L1 -> L2 mapping
    hierarchy_map_l1_l2 = build_hierarchy_map_air(
        lower_labels=l1_ids, upper_labels=l2_ids
    )

    # L2 -> L3 mapping (deduplicate)
    l2_to_l3_unique_map: Dict[int, int] = {}
    for l2_id, l3_id in zip(l2_ids, l3_ids):
        l2_to_l3_unique_map[l2_id] = l3_id

    unique_l2_list = list(l2_to_l3_unique_map.keys())
    unique_l3_list = [l2_to_l3_unique_map[l2_id] for l2_id in unique_l2_list]

    hierarchy_map_l2_l3 = build_hierarchy_map_air(
        lower_labels=unique_l2_list, upper_labels=unique_l3_list
    )

    return [hierarchy_map_l1_l2, hierarchy_map_l2_l3]


# ========================= Places365 ====================================
def get_places_fine_class_names(cat_file: str) -> List[str]:
    """Extract class names from categories_places365.txt and apply CLIP template."""
    class_names = [""] * 365
    with open(cat_file, 'r') as f:
        for line in f:
            name, fine_id = line.strip().split()
            fine_id = int(fine_id)
            clean_name = name.split('/')[-1].replace('_', ' ')
            templated_label = f"a photo of a {clean_name}"
            class_names[fine_id] = templated_label
    return class_names


def Prototype_initialization_places365():
    data_root = '/data/xiaomeng/dataset/places365/datasets/benjaminkz/places365/versions/1/'
    cat_file = os.path.join(data_root, 'categories_places365.txt')

    fine_class_names = get_places_fine_class_names(cat_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(fine_class_names, device)

    MODEL_EMBED_DIM = 384
    CLIP_EMBED_DIM = fine_clip_prototypes.shape[-1]

    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False).to('cpu')
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes.to(device)

    return fine_clip_embeds


# ========================= Things ====================================
def Prototype_initialization_things():
    """Extract 1854 base concept words and initialize with CLIP."""
    concepts_file = '/data/xiaomeng/dataset/things/02_object-level/_concepts-metadata_things.tsv'
    class_names = []

    with open(concepts_file, 'r', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        word_idx = header.index('Word')
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > word_idx:
                clean_word = parts[word_idx].strip().replace('_', ' ')
                class_names.append(f"a photo of a {clean_word}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(class_names, device)

    MODEL_EMBED_DIM = 384
    CLIP_EMBED_DIM = fine_clip_prototypes.shape[-1]

    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False).to('cpu')
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes.to(device)

    return fine_clip_embeds


# ========================= Indoor67 ====================================
def Prototype_initialization_indoor67():
    """Initialize prototypes for Indoor67 using CLIP."""
    class_names = []
    for class_name in INDOOR67_CLASSES:
        clean_name = class_name.replace('_', ' ')
        class_names.append(f"a photo of a {clean_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(class_names, device)

    MODEL_EMBED_DIM = 384
    CLIP_EMBED_DIM = fine_clip_prototypes.shape[-1]

    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False).to('cpu')
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes.to(device)

    return fine_clip_embeds


# ========================= SUN397 ====================================
classname_txt = "/data/xiaomeng/dataset/SUN397/Partitions/ClassName.txt"
excel_file = "/data/xiaomeng/dataset/SUN397/hierarchy_three_levels/three_levels.xlsx"
mapping_dict = get_sun397_mapping(classname_txt, excel_file)


def build_hierarchy_map_sun397(lower_labels: List[int], upper_labels: List[int]) -> Dict[int, List[int]]:
    hierarchy_map: Dict[int, Set[int]] = {}
    for lower_id, upper_id in zip(lower_labels, upper_labels):
        if upper_id not in hierarchy_map:
            hierarchy_map[upper_id] = set()
        hierarchy_map[upper_id].add(lower_id)
    return {u_id: sorted(list(l_ids)) for u_id, l_ids in hierarchy_map.items()}


def hierarchy_map_sun397(mapping_dict: dict = mapping_dict) -> List[Dict[int, List[int]]]:
    """
    Return three-level mapping [L1->L2, L2->L3].
    Force initialization of all 16 mid-level and 3 coarse-level keys.
    """
    hierarchy_map_l1_l2 = {i: [] for i in range(16)}
    hierarchy_map_l2_l3 = {i: [] for i in range(3)}

    mid_to_coarse = {}
    sorted_items = sorted(mapping_dict.values(), key=lambda x: x[0])

    for f_id, m_id, c_id in sorted_items:
        if f_id not in hierarchy_map_l1_l2[m_id]:
            hierarchy_map_l1_l2[m_id].append(f_id)
        mid_to_coarse[m_id] = c_id

    for m_id, c_id in mid_to_coarse.items():
        if m_id not in hierarchy_map_l2_l3[c_id]:
            hierarchy_map_l2_l3[c_id].append(m_id)

    hierarchy_map_l1_l2 = {k: sorted(v) for k, v in hierarchy_map_l1_l2.items()}
    hierarchy_map_l2_l3 = {k: sorted(v) for k, v in hierarchy_map_l2_l3.items()}

    return [hierarchy_map_l1_l2, hierarchy_map_l2_l3]


def Prototype_initialization_sun397(mapping_dict: dict = mapping_dict):
    """Extract 397 scene categories and initialize with CLIP."""
    ordered_classes = [""] * 397
    for class_name, (f_id, _, _) in mapping_dict.items():
        clean_name = class_name.split('/')[-1].replace('_', ' ')
        ordered_classes[f_id] = f"a photo of a {clean_name}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(ordered_classes, device)

    MODEL_EMBED_DIM = 384
    CLIP_EMBED_DIM = fine_clip_prototypes.shape[-1]

    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False).to('cpu')
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes.to(device)

    return fine_clip_embeds


# ========================= SUN397_2 ====================================
classname_txt = "/data/xiaomeng/dataset/SUN397/Partitions/ClassName.txt"
excel_file = "/data/xiaomeng/dataset/SUN397/hierarchy_three_levels/three_levels.xlsx"
mapping_dict_2 = get_sun397_2_mapping(classname_txt, excel_file)


def hierarchy_map_sun397_2(mapping_dict: dict = mapping_dict_2) -> List[Dict[int, List[int]]]:
    """Return two-level mapping (16 -> 3)."""
    hierarchy_map_mid_coarse = {i: [] for i in range(3)}

    mid_to_coarse = {}
    for mid_id, coarse_id in mapping_dict.values():
        mid_to_coarse[mid_id] = coarse_id

    for m_id, c_id in mid_to_coarse.items():
        if m_id not in hierarchy_map_mid_coarse[c_id]:
            hierarchy_map_mid_coarse[c_id].append(m_id)

    hierarchy_map_mid_coarse = {k: sorted(v) for k, v in hierarchy_map_mid_coarse.items()}
    return [hierarchy_map_mid_coarse]


def Prototype_initialization_sun397_2(mapping_dict: dict = mapping_dict):
    """Initialize 16 mid-level prototypes using CLIP."""
    mid_class_names = [
        "shopping and dining", "workplace", "home or hotel", "indoor transportation",
        "indoor sports and leisure", "indoor cultural", "water, ice, snow",
        "mountains, hills, desert, sky", "forest, field, jungle", "man-made elements",
        "outdoor transportation", "cultural or historical building",
        "sports fields, parks, leisure spaces", "industrial and construction",
        "houses, cabins, gardens, and farms", "commercial buildings, shops, markets, cities, and towns"
    ]

    prompts = [f"a photo of a {name}" for name in mid_class_names]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(prompts, device)

    MODEL_EMBED_DIM = 384
    CLIP_EMBED_DIM = fine_clip_prototypes.shape[-1]

    if CLIP_EMBED_DIM != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(CLIP_EMBED_DIM, MODEL_EMBED_DIM, bias=False).to('cpu')
        with torch.no_grad():
            fine_prototypes_projected = projection_layer(fine_clip_prototypes.to('cpu'))
        fine_clip_embeds = fine_prototypes_projected.to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes.to(device)

    return fine_clip_embeds


# ========================= MiniPlaces ====================================
data_root = '/data/xiaomeng/dataset/miniplaces/'
io_txt = '/data/xiaomeng/dataset/places365/datasets/benjaminkz/places365/versions/1/IO_places365.txt'
train_dir = os.path.join(data_root, 'images', 'train')

mapping_dict, ordered_names = MiniPlaces.get_miniplaces_mapping_from_dir(train_dir, io_txt)


def hierarchy_map_miniplaces(mapping_dict: dict = mapping_dict) -> List[Dict[int, List[int]]]:
    """Return 100 -> 2 level hierarchy mapping."""
    hierarchy_map_fine_coarse = {0: [], 1: []}
    sorted_items = sorted(mapping_dict.values(), key=lambda x: x[0])
    for f_id, c_id in sorted_items:
        hierarchy_map_fine_coarse[c_id].append(f_id)
    hierarchy_map_fine_coarse = {k: sorted(v) for k, v in hierarchy_map_fine_coarse.items()}
    return [hierarchy_map_fine_coarse]


def Prototype_initialization_miniplaces(ordered_names: list = ordered_names):
    """Generate CLIP prototypes (100 x 384) for MiniPlaces."""
    class_prompts = []
    for name in ordered_names:
        clean_name = name.split('/')[-1].replace('_', ' ')
        class_prompts.append(f"a photo of a {clean_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fine_clip_prototypes = load_clip_and_encode_text(class_prompts, device)

    MODEL_EMBED_DIM = 384
    if fine_clip_prototypes.shape[-1] != MODEL_EMBED_DIM:
        print("\nWarning: CLIP dimension mismatch. Adding projection layer.")
        projection_layer = nn.Linear(fine_clip_prototypes.shape[-1], MODEL_EMBED_DIM, bias=False).to('cpu')
        with torch.no_grad():
            fine_clip_embeds = projection_layer(fine_clip_prototypes.to('cpu')).to(device)
    else:
        fine_clip_embeds = fine_clip_prototypes.to(device)

    return fine_clip_embeds