# import os
# import kagglehub

# os.environ["KAGGLEHUB_CACHE"] = "/data/xiaomeng/dataset/places365"
# path = kagglehub.dataset_download("benjaminkz/places365")

# print("Successfully loaded:", path)

import os
import json


def build_places365_tree(places_root, output_path):
    """
    Build the Places365 hierarchical tree file.

    Fine Level: 365
    Coarse Level: 2 (0: Indoor, 1: Outdoor)
    """
    cat_file = os.path.join(places_root, 'categories_places365.txt')
    io_file = os.path.join(places_root, 'IO_places365.txt')

    # Map name to fine_id
    name_to_fine = {}
    with open(cat_file, 'r', encoding='utf-8') as f:
        for line in f:
            name, fine_id = line.strip().split()
            name_to_fine[name] = int(fine_id)

    # Map name to coarse_id and build [fine_id, coarse_id] pairs
    tree_list = []
    with open(io_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            name = parts[0]
            # IO file uses 1 (Indoor) and 2 (Outdoor) -> convert to 0 and 1
            coarse_id = int(parts[1]) - 1

            if name in name_to_fine:
                fine_id = name_to_fine[name]
                tree_list.append([fine_id, coarse_id])

    # Sort by fine_id for consistency
    tree_list.sort(key=lambda x: x[0])

    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_list, f)
    print(f"[*] Places365 tree generated: {output_path} ({len(tree_list)} nodes)")


def build_things_tree(things_root, output_path):
    """
    Build the THINGS hierarchical tree file.

    Fine Level: 1854
    Coarse Level: 27
    """
    category_file = os.path.join(things_root, '03_category-level/category27_manual.tsv')

    tree_list = []
    with open(category_file, 'r', encoding='utf-8') as f:
        # Skip the header row containing the 27 category names
        _ = f.readline()

        fine_id = 0
        for line in f:
            parts = line.strip().split('\t')
            try:
                # Find which column has '1' indicating the coarse category
                coarse_id = parts.index('1')
                tree_list.append([fine_id, coarse_id])
            except ValueError:
                # If the row is all zeros (fallback), assign to class 0
                tree_list.append([fine_id, 0])
            fine_id += 1

    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_list, f)
    print(f"[*] THINGS tree generated: {output_path} ({len(tree_list)} nodes)")


def build_things_plus_tree(things_root, output_path):
    """
    Build the THINGS-PLUS hierarchical tree file.

    Fine Level: 1854
    Coarse Level: 53
    """
    category_file = os.path.join(things_root, '03_category-level/category53_wide-format.tsv')

    tree_list = []
    with open(category_file, 'r', encoding='utf-8') as f:
        # Skip header row
        _ = f.readline()

        fine_id = 0
        for line in f:
            parts = line.strip().split('\t')

            # Skip the first two text columns (uniqueID and Word); keep only the 0/1 columns
            coarse_parts = parts[2:]

            try:
                # Find the index of '1' in the 0/1 list, giving the coarse_id from 0-52
                coarse_id = coarse_parts.index('1')

                # Safety check: if index exceeds 52, default to class 0
                if coarse_id >= 53:
                    coarse_id = 0

                tree_list.append([fine_id, coarse_id])
            except ValueError:
                # Handle unassigned categories (all zeros)
                tree_list.append([fine_id, 0])

            fine_id += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_list, f)

    print(f"[*] THINGS-PLUS tree generated: {output_path} ({len(tree_list)} nodes)")


if __name__ == '__main__':
    things_root = '/data/xiaomeng/dataset/things/'
    output_json_path = '/data/xiaomeng/model/HCAST-V1/HCAST-V2/data/things_plus_tree.json'

    build_things_plus_tree(things_root, output_json_path)