import os
import json
import pandas as pd


def build_sun397_tree(classname_txt, excel_file, output_path):
    """
    Build a SUN397 hierarchical tree file.

    Output format: [[fine_id, mid_id, coarse_id], ...]  (397 entries)
    """
    print("Building SUN397 hierarchical tree (397 -> 16 -> 3) ...")

    # Map class path (e.g., "/a/abbey") to fine_id (0-396)
    name_to_id = {}
    if not os.path.exists(classname_txt):
        print(f"Error: file not found {classname_txt}")
        return

    with open(classname_txt, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            class_path = line.strip()
            name_to_id[class_path] = idx

    # Read Excel file; header=1 skips the top-level heading row
    try:
        df = pd.read_excel(excel_file, header=1)
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return

    tree_list = []
    processed_fine_ids = set()

    for _, row in df.iterrows():
        # Extract class name from first column (remove any extra quotes)
        cell_val = str(row.iloc[0]).strip()
        clean_name = cell_val.replace("'", "")

        # Skip empty or invalid rows
        if not clean_name.startswith("/") or clean_name not in name_to_id:
            continue

        fine_id = name_to_id[clean_name]
        if fine_id in processed_fine_ids:
            continue

        # Coarse ID: columns 1-3 correspond to indoor, outdoor natural, outdoor man-made
        coarse_id = -1
        if row.iloc[1] == 1:
            coarse_id = 0
        elif row.iloc[2] == 1:
            coarse_id = 1
        elif row.iloc[3] == 1:
            coarse_id = 2

        # Mid ID: columns 4-20 (16 sub-categories)
        mid_id = -1
        for i in range(4, 21):
            if i < len(row) and row.iloc[i] == 1:
                mid_id = i - 4
                break

        if coarse_id != -1 and mid_id != -1:
            tree_list.append([fine_id, mid_id, coarse_id])
            processed_fine_ids.add(fine_id)

    # Sort by fine_id to align with label order
    tree_list.sort(key=lambda x: x[0])

    if len(tree_list) != 397:
        print(f"[!] Warning: extracted {len(tree_list)}/397 classes. "
              "Check that class names in ClassName.txt match the Excel first column exactly (including slashes).")
    else:
        print("[*] Successfully extracted full 397-class hierarchy.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_list, f)

    print(f"[*] Tree file saved to: {output_path}")


def build_sun397_2_tree(classname_txt, excel_file, output_path):
    """
    Build a SUN397 tree file containing only the 16 -> 3 mapping.

    Output format: [[mid_id, coarse_id], ...] (16 unique entries)
    """
    print("Building SUN397 hierarchical tree (16 -> 3) ...")

    # Valid class names from ClassName.txt
    name_to_valid = set()
    if os.path.exists(classname_txt):
        with open(classname_txt, 'r', encoding='utf-8') as f:
            for line in f:
                name_to_valid.add(line.strip())

    try:
        df = pd.read_excel(excel_file, header=1)
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return

    tree_set = set()  # Use set to deduplicate

    for _, row in df.iterrows():
        clean_name = str(row.iloc[0]).strip().replace("'", "")

        # Match class name (with or without leading slash)
        if clean_name not in name_to_valid:
            alt_name = clean_name[1:] if clean_name.startswith('/') else '/' + clean_name
            if alt_name not in name_to_valid:
                continue

        # Coarse ID
        coarse_id = 0
        if row.iloc[1] == 1:
            coarse_id = 0
        elif row.iloc[2] == 1:
            coarse_id = 1
        elif row.iloc[3] == 1:
            coarse_id = 2

        # Mid ID
        mid_id = 0
        for i in range(4, 21):
            if i < len(row) and row.iloc[i] == 1:
                mid_id = i - 4
                break

        tree_set.add((mid_id, coarse_id))

    tree_list = [list(x) for x in tree_set]
    tree_list.sort(key=lambda x: x[0])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(tree_list, f)

    print(f"[*] Tree file saved to: {output_path} ({len(tree_list)} unique mappings)")


if __name__ == '__main__':
    data_root = '/data/xiaomeng/dataset/SUN397/'
    classname_txt = os.path.join(data_root, 'Partitions', 'ClassName.txt')
    excel_file = os.path.join(data_root, 'hierarchy_three_levels', 'three_levels.xlsx')

    output_path = 'data/sun397_2_tree.json'

    print("Starting SUN397 hierarchical tree construction ...")
    build_sun397_2_tree(classname_txt, excel_file, output_path)