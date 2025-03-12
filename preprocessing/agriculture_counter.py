import cv2
import os
import shutil
from scipy.ndimage import label

root_path = "../../Dataset/LoveDA/original/val/urban"
image_dir = os.path.join(root_path, "images_png")
ground_truth_dir = os.path.join(root_path, "masks_agriculture")

for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)
    ground_truth_path = os.path.join(ground_truth_dir, image_name)

    print(f"Processing image: {image_name}")

    ground_truth_image = cv2.imread(ground_truth_path, cv2.IMREAD_GRAYSCALE)

    _, num_areas = label(ground_truth_image)

    num_areas_path = os.path.join(root_path, str(num_areas))
    output_image_path = os.path.join(num_areas_path, "images_png")
    output_gt_path = os.path.join(num_areas_path, "masks_agriculture")

    os.makedirs(num_areas_path, exist_ok=True)
    os.makedirs(output_image_path, exist_ok=True)
    os.makedirs(output_gt_path, exist_ok=True)

    shutil.copy(image_path, f"{output_image_path}/{image_name}")
    shutil.copy(ground_truth_path, f"{output_gt_path}/{image_name}")

