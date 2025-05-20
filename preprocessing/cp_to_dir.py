import os
import shutil

dir_source = "../../Dataset/LoveDA/cleansing_data/test/images_png"
dir_search = "../../Dataset/LoveDA/original/train/rural/masks_png"
# dir_search = "../../Dataset/LoveDA/original/train/urban/masks_png"
# dir_search = "../../Dataset/LoveDA/original/val/rural/masks_png"
# dir_search = "../../Dataset/LoveDA/original/val/urban/masks_png"
dir_target = "../../Dataset/LoveDA/cleansing_data/test/all_masks_png"

os.makedirs(dir_target, exist_ok=True)

for filename in os.listdir(dir_source):
    if filename.endswith(".png"):
        src_file = os.path.join(dir_search, filename)
        dst_file = os.path.join(dir_target, filename)

        if os.path.exists(src_file):
            shutil.copy(src_file, dst_file)
            print(f"Copied: {filename}")
        else:
            print(f"File not found: {filename}")
