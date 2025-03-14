import os
import re
import shutil

main_directory = "../../Dataset/LoveDA/original/val/urban/agriculture_area"
target_dir = "../../Dataset/LoveDA/agriculture_area"

folders = []

for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)

    if not os.path.isdir(folder_path) or not folder.isdigit():
        continue

    folders.append(folder_path)

sorted_folders = sorted(folders, key=lambda x: int(re.search(r'/(\d+)$', x).group(1)))

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(target_dir, split), exist_ok=True)

for folder in sorted_folders:
    src_images_png = os.path.join(folder, "images_png")
    src_masks_png = os.path.join(folder, "masks_agriculture")
    images_png = sorted(os.listdir(src_images_png)) 
    masks_png = sorted(os.listdir(src_masks_png))
    
    total_images = len(images_png)
    num_train = int(total_images * 0.8)
    num_val = int(total_images * 0.1)
    num_test = total_images - num_train - num_val

    splits = {
        "train": images_png[:num_train],
        "val": images_png[num_train:num_train + num_val],
        "test": images_png[num_train + num_val:]
    }

    for split, img_list in splits.items():
        root_dst = os.path.join(target_dir, split)
        dst_images_png = os.path.join(root_dst, "images_png")
        dst_masks_png = os.path.join(root_dst, "masks_png")

        os.makedirs(dst_images_png, exist_ok=True)
        os.makedirs(dst_masks_png, exist_ok=True)
        
        for img in img_list:
            src_path_images = os.path.join(src_images_png, img)
            dst_path_images = os.path.join(dst_images_png, img)
            shutil.copy(src_path_images, dst_images_png)

            src_path_masks = os.path.join(src_masks_png, img)
            dst_path_masks = os.path.join(dst_masks_png, img)
            shutil.copy(src_path_masks, dst_masks_png)

    print(f"✅ {folder}: {num_train} train, {num_val} val, {num_test} test")

print("📂 Dataset telah dipisah ke train, val, dan test tanpa subfolder kategori, dalam urutan asli.")