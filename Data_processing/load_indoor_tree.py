import os
import json

# 请将此处替换为你的 TrainImages.txt 路径
txt_file = '/data/xiaomeng/dataset/indoorCVPR/TrainImages.txt'
output_path = 'data/indoor67_tree.json'

INDOOR67_SUPER_DICT = {
    'bakery': 0, 'bookstore': 0, 'clothingstore': 0, 'deli': 0, 'florists': 0, 
    'grocerystore': 0, 'hairsalon': 0, 'jewelryshop': 0, 'mall': 0, 'shoeshop': 0, 
    'toystore': 0, 'videostore': 0,
    'bathroom': 1, 'bedroom': 1, 'children_room': 1, 'closet': 1, 'dining_room': 1, 
    'kitchen': 1, 'livingroom': 1, 'nursery': 1, 'pantry': 1,
    'airport_inside': 2, 'artstudio': 2, 'auditorium': 2, 'cloister': 2, 'corridor': 2, 
    'elevator': 2, 'escalator': 2, 'greenhouse': 2, 'hospitalroom': 2, 'inside_bus': 2, 
    'inside_subway': 2, 'laundromat': 2, 'library': 2, 'lobby': 2, 'museum': 2, 
    'operating_room': 2, 'prisoncell': 2, 'stairscase': 2, 'subway': 2, 'trainstation': 2, 
    'waitingroom': 2,
    'bar': 3, 'bowling': 3, 'buffet': 3, 'casino': 3, 'church_inside': 3, 'cinema': 3, 
    'concert_hall': 3, 'fastfood_restaurant': 3, 'gameroom': 3, 'gym': 3, 
    'movietheater': 3, 'poolinside': 3, 'restaurant': 3, 'restaurant_kitchen': 3, 
    'winecellar': 3,
    'classroom': 4, 'computerroom': 4, 'dentaloffice': 4, 'laboratory': 4, 
    'meeting_room': 4, 'office': 4, 'studio': 4, 'tv_studio': 4, 'locker_room': 4,
    'kindergarden': 4
}

unique_classes = []
with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        path = line.strip()
        if not path: continue
        class_name = path.split('/')[0] 
        if class_name not in unique_classes:
            unique_classes.append(class_name)

unique_classes.sort()

tree_list = []
for fine_id, class_name in enumerate(unique_classes):
    coarse_id = INDOOR67_SUPER_DICT.get(class_name, 2)
    tree_list.append([fine_id, coarse_id])

os.makedirs('data', exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(tree_list, f)
print(f"[*] Indoor67 Tree generated: {output_path} total {len(tree_list)} nodes")