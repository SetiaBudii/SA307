import os
import shutil

image_folder = (
    "../../Dataset/LoveDA/val/masks_png"
)

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    source = f"../../Dataset/LoveDA/val/images_png/{image_name}"
    destination = f"../../Dataset/LoveDA/agriculture/val/images_png/{image_name}"

    shutil.copy(source, destination)
