import numpy as np
from PIL import Image
import json
from pycocotools import mask as mask_util
import os
from tqdm import tqdm
import cv2
import os
import torch
from scipy.spatial.distance import cdist
from scipy.ndimage import center_of_mass, distance_transform_edt, label

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


def read_single(data):
    ent  = data[np.random.randint(len(data))]
    Img = cv2.imread(ent["image"])[...,::-1]
    ann_map = cv2.imread(ent["annotation"])

    mat_map = ann_map[:,:,0]
    ves_map = ann_map[:,:,2]
    mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1)

    inds = np.unique(mat_map)[1:]
    if inds.__len__()>0:
            ind = inds[np.random.randint(inds.__len__())]
    else:
            return read_single(data)

    mask=(mat_map == ind).astype(np.uint8)
    coords = np.argwhere(mask > 0)
    yx = np.array(coords[np.random.randint(len(coords))])
    return Img,mask,[[yx[1], yx[0]]]

def read_batch(data,batch_size=4):
    limage = []
    lmask = []
    linput_point = []
    for i in range(batch_size):
            image,mask,input_point = read_single(data)
            limage.append(image)
            lmask.append(mask)
            linput_point.append(input_point)

    return limage, np.array(lmask), np.array(linput_point),  np.ones([batch_size,1])


def append_data(data_dir, data_dir_mask):
    for name in os.listdir(data_dir):  # Iterate over all files in data_dir
        if name.endswith(".png"):  # Process only files with .png extension
            image_path = os.path.join(data_dir, name)  # Full path to image
            annotation_path = os.path.join(data_dir_mask, name)  # Full path to mask
            
            # Check if both image and mask exist
            if os.path.exists(image_path) and os.path.exists(annotation_path):
                data.append({"image": image_path, "annotation": annotation_path})
            else:
                print(f"Warning: Missing mask for image '{name}' or invalid paths.")


def read_single_center(data):
    img = cv2.cvtColor(cv2.imread(data["image"]), cv2.COLOR_BGR2RGB)
    gt_img = cv2.imread(data["annotation"], cv2.IMREAD_GRAYSCALE)

    input_points = []
    input_labels = []

    agriculture = (gt_img == 7).astype(np.uint8)

    labeled_array, num_instances = label(agriculture)
    instances = [(labeled_array == i).astype(np.uint8) for i in range(1, num_instances + 1)]
    pixel_counts = [np.sum(instance) for instance in instances]

    max_index = int(np.argmax(pixel_counts))
    largest_instance = instances[max_index]

    centroid_y, centroid_x = center_of_mass(largest_instance)

    dist_map = distance_transform_edt(largest_instance)
    threshold = 20
    candidate_coords = np.column_stack(np.where(dist_map >= threshold))
    if len(candidate_coords) == 0:
        foreground_coords = np.column_stack(np.where(largest_instance == 1))
        distances = cdist([(centroid_y, centroid_x)], foreground_coords)
        nearest_idx = np.argmin(distances)
        centroid_y, centroid_x = foreground_coords[nearest_idx]
    else:
        distances = cdist([(centroid_y, centroid_x)], candidate_coords)
        nearest_idx = np.argmin(distances)
        centroid_y, centroid_x = candidate_coords[nearest_idx]

    first_point = [centroid_x, centroid_y]
    input_points.append(first_point)
    input_points = np.array(input_points)
    input_labels = np.ones(len(input_points), dtype=int)

    gt_img = torch.from_numpy(gt_img)
    gt_img = (gt_img == 7).float()
    gt_img = gt_img.unsqueeze(0).cuda()

    return img, gt_img, input_points, input_labels