import numpy as np
from PIL import Image
import os
from pycocotools import mask as mask_util


def ground_truth_to_json(image_path, image_id, file_name):
    # 1. Load image
    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img_array = np.array(img)
    height, width = img_array.shape

    # 2. Calculate area
    unique_values, counts = np.unique(img_array, return_counts=True)
    areas = dict(zip(unique_values, counts))  # Area per grayscale value

    # 3. Create RLE and JSON annotations
    annotations = []
    for gray_value in unique_values:
        if gray_value == 0:
            continue  # Skip background

        # Create binary mask for the current grayscale value
        binary_mask = (img_array == gray_value).astype(np.uint8)

        # Get RLE
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("utf-8")  # Convert to string

        # Annotation for this level
        annotation = {
            "area": int(areas[gray_value]),
            "segmentation": {
                "size": [height, width],
                "counts": rle["counts"],
            },
            "grayscale_value": int(gray_value),
        }
        if gray_value == 7:
            annotations.append(annotation)

    # 4. Create JSON
    output_json = {
        "image": {
            "image_id": image_id,
            "width": width,
            "height": height,
            "file_name": file_name,
        },
        "annotations": annotations,
    }

    return output_json

def rle_to_mask(rle, size):
    # Decode RLE encoded segmentation to mask
    return mask_util.decode(rle).reshape(size)

# MAIN CODE
# Folder yang berisi gambar ground truth
image_folder = (
    "../../Dataset/LoveDA/original/val/urban/masks_png"
)

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    print(f"Processing image: {image_name}")

    image_id = image_name.split(".")[0]
    json_output = ground_truth_to_json(image_path, image_id, image_name)

    image_info = json_output["image"]
    annotations = json_output["annotations"]

    width, height = image_info["width"], image_info["height"]

    gt_image = np.zeros((height, width), dtype=np.uint8)

    for i, annotation in enumerate(annotations):
        segmentation = annotation["segmentation"]

        rle = {"size": segmentation["size"], "counts": segmentation["counts"]}
        mask = rle_to_mask(rle, (height, width))

        gt_image[mask > 0] = annotation["grayscale_value"]

    gt_image_pil = Image.fromarray(gt_image)
    # CEK PATH SEBELUM RUN CODE
    gt_image_pil.save(f"../../Dataset/LoveDA/original/val/urban/masks_agriculture/{image_info['file_name']}")

