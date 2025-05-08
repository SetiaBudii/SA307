import random
import os
import sys
sys.path.insert(0, '..')
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import label
from npc.newnpc import waterbodies_extraction


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
    image_path,
):
    water_bodies = waterbodies_extraction(image_path)

    labeled_array, num_features = label(water_bodies)
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
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(gt, cmap='gray')
        axes[0].set_title('Init Result')
        axes[0].axis('off')
        
        axes[1].imshow(pred, cmap='gray')
        axes[1].set_title(f'Water {i}')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Intersection')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

        if iou > 0.1:
            neg_random_point = neg_coords[np.random.randint(len(neg_coords))]
            neg_points.append([neg_random_point[1], neg_random_point[0]])
            neg_labels = np.zeros(len(neg_points), dtype=int)

    return neg_points, neg_labels