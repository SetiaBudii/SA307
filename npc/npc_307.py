import random
import os
import sys
sys.path.insert(0, '..')
import torch
import numpy as np
import matplotlib.pyplot as plt
from npc.hsv_negative_prompt_augmentation import get_mask
import cv2
from scipy.ndimage import center_of_mass, label

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou

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

def calc_iou_matrix(mask_list1, mask_list2):
    iou_matrix = torch.zeros((len(mask_list1), len(mask_list2)))
    for i, mask1 in enumerate(mask_list1):
        for j, mask2 in enumerate(mask_list2):
            iou_matrix[i, j] = calculate_iou(mask1, mask2)
    return iou_matrix

def cal_mask_ious(
    cfg,
    model,
    images_weak,
    prompts,
    gt_masks,
):
    with torch.no_grad():
        _, soft_masks, _, _ = model(images_weak, prompts)   

    for i, (soft_mask, gt_mask) in enumerate(zip(soft_masks, gt_masks)):  
        soft_mask = (soft_mask > 0).float()
        mask_ious = calc_iou_matrix(soft_mask, soft_mask)
        indices = torch.arange(mask_ious.size(0))
        mask_ious[indices, indices] = 0.0
    return mask_ious, soft_mask


def neg_prompt_calibration(
    masks,
    image,
):
    neg_points = np.array([])  # atau bentuk array kosong sesuai format yang diharapkan
    neg_labels = np.array([])
    masks_candidate = get_mask(image)

    labeled_array, num_features = label(masks_candidate)
    individual_features = [(labeled_array == i).astype(np.uint8) for i in range(1, num_features + 1)]
    
    neg_points = []
    for i in range(1, num_features + 1):
        iou = calc_iou(masks[0], individual_features[i-1])
        neg_coords = np.argwhere(individual_features[i-1] > 0)\
        
        gt = masks[0]
        pred = individual_features[i-1]
        
        gt_bool = gt.astype(bool)
        pred_bool = pred.astype(bool)
        
        overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
        overlay[gt_bool] = [255, 195, 128]
        overlay[pred_bool] = [0, 0, 255]
        overlay[gt_bool & pred_bool] = [255, 255, 0]
        
        if iou > 0.005:
            neg_random_point = neg_coords[np.random.randint(len(neg_coords))]
            neg_points.append([neg_random_point[1], neg_random_point[0]])
            neg_labels = np.zeros(len(neg_points), dtype=int)

    return neg_points, neg_labels


def npc(gt_path, init_mask, class_value, type):
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        class_masks = gt_img * (gt_img == class_value)
        labeled_array, num_instances = label(class_masks)
        instances = [(labeled_array == i).astype(np.uint8) for i in range(1, num_instances + 1)]

        neg_points = []
        neg_labels = []
        for i in range(1, num_instances + 1):
            iou = calc_iou(init_mask, instances[i-1])
            neg_coords = np.argwhere(instances[i-1] > 0)

            instance = instances[i-1]

            init_mask_bool = init_mask.astype(bool)
            instance_bool = instance.astype(bool)

            if iou > 0.005:
                intersection_mask = init_mask_bool & instance_bool

                if type == 1: # random
                    neg_coords = np.argwhere(intersection_mask)
                    neg_random_point = neg_coords[np.random.randint(len(neg_coords))]
                    neg_points.append([neg_random_point[1], neg_random_point[0]])
                elif type == 2: # center
                    centroid_y, centroid_x = center_of_mass(intersection_mask)
                    neg_points.append([centroid_x, centroid_y])

                neg_labels = np.zeros(len(neg_points), dtype=int)

        return np.array(neg_points), np.array(neg_labels)
        

def npc_hsv(masks, image, num_points):    
    neg_points = np.array([])  # atau bentuk array kosong sesuai format yang diharapkan
    neg_labels = np.array([])

    if (num_points > 0):
        masks_candidate = get_mask(image)

        labeled_array, num_features = label(masks_candidate)
        individual_features = [(labeled_array == i).astype(np.uint8) for i in range(1, num_features + 1)]
        
        neg_points = []
        for i in range(1, num_features + 1):
            iou = calc_iou(masks[0], individual_features[i-1])
            neg_coords = np.argwhere(individual_features[i-1] > 0)

            gt = masks[0]
            pred = individual_features[i-1]
            
            gt_bool = gt.astype(bool)
            pred_bool = pred.astype(bool)
            
            overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
            overlay[gt_bool] = [255, 195, 128]
            overlay[pred_bool] = [0, 0, 255]
            overlay[gt_bool & pred_bool] = [255, 255, 0]
            
            if iou > 0.005:
                neg_random_point = neg_coords[np.random.randint(len(neg_coords))]
                neg_points.append([neg_random_point[1], neg_random_point[0]])
                neg_labels = np.zeros(len(neg_points), dtype=int)

    return neg_points, neg_labels