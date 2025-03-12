import os
import cv2
import numpy as np

# Folder input (ground truth grayscale mask)
image_folder = "../../Dataset/LoveDA/original/train/urban/masks_png"

# Folder output untuk color mask
output_folder = "../../Dataset/LoveDA/original/train/urban/masks_color"

os.makedirs(output_folder, exist_ok=True)

def grayscale_to_color(image):
    color_map = {
        1: (255, 255, 255),
        2: (255, 0, 0),
        3: (255, 255, 0),
        4: (0, 0, 255),
        5: (159, 129, 183),
        6: (0, 255, 0),
        7: (255, 195, 128)
    }

    h, w = image.shape
    color_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for gray_value, color in color_map.items():
        mask = image == gray_value
        color_image[mask] = color
    
    return color_image

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    output_path = os.path.join(output_folder, image_name)

    print(f"Processing image: {image_name}")

    grayscale_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if grayscale_mask is None:
        print(f"Warning: Gagal membaca {image_name}, dilewati.")
        continue

    color_mask = grayscale_to_color(grayscale_mask)
    rgb_mask = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

    cv2.imwrite(output_path, rgb_mask)

print("Proses konversi selesai!")
