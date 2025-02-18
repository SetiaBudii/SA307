import numpy as np
from PIL import Image
import json
from pycocotools import mask as mask_util
import os
from tqdm import tqdm

def ground_truth_to_json(image_path, image_id, file_name):
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)
    height, width = img_array.shape

    unique_values, counts = np.unique(img_array, return_counts=True)
    areas = dict(zip(unique_values, counts))

    annotations = []
    for gray_value in unique_values:
        if gray_value == 0:
            continue

        binary_mask = (img_array == gray_value).astype(np.uint8)
        rle = mask_util.encode(np.asfortranarray(binary_mask))
        rle["counts"] = rle["counts"].decode("utf-8")

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


def all_image_loveda_to_json(image_dir, output_json_path):
    image_urban = image_dir + "/urban/masks_png"
    image_rural = image_dir + "/rural/masks_png"
    image_dirs = [image_urban, image_rural]

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    for images in image_dirs:
        images_files = [f for f in os.listdir(images) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

        for image_name in tqdm(image_files, desc=f"Processing {images}", unit="file"):
            image_path = os.path.join(images, image_name)

            image_id = os.path.splitext(image_name)[0]
            json_output = ground_truth_to_json(image_path, image_id, image_name)

            json_file_name = f"{image_id}.json"
            json_file_path = os.path.join(output_json_path, json_file_name)
            with open(json_file_path, "w") as f:
                json.dump(json_output, f, indent=4)
    print("All images processed.")


def rle_to_mask(rle, size):
    return mask_utils.decode(rle).reshape(size)


def all_json_to_image(json_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    json_files = [f for f in os.listdir(json_dir) if f.lower().endswith(".json")]

    for json_file in tqdm(json_files, desc="Processing JSON files", unit="file"):
        input_json_path = os.path.join(input_folder, json_file)
        output_image_path = os.path.join(output_folder, json_file.replace(".json", ".png"))

        with open(input_json_path, "r") as f:
            data = json.load(f)

        if "image" not in data or "annotations" not in data:
            print(f"File {json_file} tidak memiliki kunci 'image' atau 'annotations'. Melewati...")
            continue

        image_info = data["image"]
        annotations = data["annotations"]
        width, height = image_info["width"], image_info["height"]
        gt_image = np.zeros((height, width), dtype=np.uint8)

        for annotation in annotations:
            segmentation = annotation["segmentation"]
            if "size" in segmentation and "counts" in segmentation:
                rle = {"size": segmentation["size"], "counts": segmentation["counts"]}
                mask = rle_to_mask(rle, (height, width))
                grayscale_value = annotation.get("grayscale_value", 1)
                gt_image[mask > 0] = grayscale_value

        gt_image_pil = Image.fromarray(gt_image)
        gt_image_pil.save(output_image_path)

    print("Proses konversi selesai untuk semua file JSON.")

