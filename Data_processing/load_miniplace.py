# import os
# import kagglehub

# # 1. 设置为你想要保存的路径
# os.environ["KAGGLEHUB_CACHE"] = "/data/xiaomeng/dataset/places365"

# # 2. 正常下载
# path = kagglehub.dataset_download("benjaminkz/places365")

# print("数据集已下载至:", path)

import os
import json
import sys
sys.path.append('/data/xiaomeng/model/HCAST-V1/HCAST-V2/deit/dataset')
sys.path.append('/data/xiaomeng/model/HCAST-V1/HCAST-V2/deit')
import MiniPlaces

def build_miniplaces_tree(mapping_dict: dict, output_path: str):
    tree_list = []
    sorted_items = sorted(mapping_dict.values(), key=lambda x: x[0])
    for f_id, c_id in sorted_items:
        tree_list.append([f_id, c_id])
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_list, f)
    print(f"[*] MiniPlaces file generated: {output_path} (total {len(tree_list)} entries)")
if __name__ == '__main__':
    # MiniPlaces
    data_root = '/data/xiaomeng/dataset/miniplaces/'
    
    io_txt = '/data/xiaomeng/dataset/places365/datasets/benjaminkz/places365/versions/1/IO_places365.txt'
    train_dir = os.path.join(data_root, 'images', 'train')
    
    mapping_dict, ordered_names = MiniPlaces.get_miniplaces_mapping_from_dir(train_dir, io_txt)
        
    output_json_path = '/data/xiaomeng/model/HCAST-V1/HCAST-V2/data/miniplaces_tree.json'
    
    build_miniplaces_tree(mapping_dict, output_json_path)