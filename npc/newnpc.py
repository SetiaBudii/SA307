import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from scipy.ndimage import convolve
from skimage.morphology import closing, remove_small_objects, disk
import torch
import cv2
from scipy.ndimage import center_of_mass, label

def waterbodies_extraction(image_path):
    img = io.imread(image_path)
    entropy_img = entropy(img[:, :, 0], footprint=disk(3))
    thresh = threshold_otsu(entropy_img)
    binary_img = entropy_img <= thresh

    # --- Tahapan post-processing ---
    # Langkah 1: Filter berdasarkan neighbor (padat/tidak bolong)
    dense_binary_img = refine_mask(binary_img, iterations=2, min_neighbors=8)

    # # Langkah 2: Haluskan tepi objek
    # dense_binary_img = closing(dense_binary_img, disk(2))

    # Langkah 3: Buang objek kecil yang tidak signifikan
    dense_binary_img = remove_small_objects(dense_binary_img, min_size=1000)

    return dense_binary_img

def refine_mask(binary_img, iterations=1, min_neighbors=6):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    result = binary_img.copy()
    for _ in range(iterations):
        neighbor_count = convolve(result.astype(np.uint8), kernel, mode='constant', cval=0)
        result = result & (neighbor_count >= min_neighbors)
    return result


def calc_iou(prd, gt):
    if isinstance(gt, torch.Tensor):
        gt = gt.squeeze(0).cpu().numpy()

    assert prd.shape == gt.shape
    prd = prd.reshape(prd.size).copy()
    gt = gt.reshape(gt.size)

    area_intersection = np.logical_and(prd, gt).sum()
    area_union = np.logical_or(prd, gt).sum()

    iou = area_intersection / (area_union + 1e-10)
    return iou


def npc(image_path, init_mask, class_value):
    gt_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if class_value == 8:
        class_masks = (8 * np.isin(gt_img, [1, 4, 5, 6])).astype(np.uint8)
    else:
        class_masks = gt_img * (gt_img == class_value)

    labeled_array, num_instances = label(class_masks)
    instances = [(labeled_array == i).astype(np.uint8) for i in range(1, num_instances + 1)]

    neg_points = []
    neg_labels = []
    for i in range(1, num_instances + 1):
        iou = calc_iou(init_mask, instances[i-1])
        neg_coords = np.argwhere(instances[i-1] > 0)

        gt = init_mask
        pred = instances[i-1]

        gt_bool = gt.astype(bool)
        pred_bool = pred.astype(bool)

        if iou > 0.005:
            intersection_mask = gt_bool & pred_bool
            neg_coords = np.argwhere(intersection_mask)

            if len(neg_coords) > 0:
                neg_random_point = neg_coords[np.random.randint(len(neg_coords))]
                neg_points.append([neg_random_point[1], neg_random_point[0]])

            neg_labels = np.zeros(len(neg_points), dtype=int)

    return np.array(neg_points), np.array(neg_labels)
