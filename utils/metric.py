import numpy as np
import torch
import json

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

def calc_metrics(prd, gt):
    # Pastikan ground truth (gt) adalah numpy array
    if isinstance(gt, torch.Tensor):
        gt = gt.squeeze(0).cpu().numpy()
    
    # Verifikasi bahwa dimensi prediksi dan ground truth sama
    assert prd.shape == gt.shape
    
    # Reshape untuk memastikan formatnya konsisten
    prd = prd.reshape(prd.size).copy()
    gt = gt.reshape(gt.size)
    
    # Hitung area intersection dan union
    area_intersection = np.logical_and(prd, gt).sum()
    area_union = np.logical_or(prd, gt).sum()

    # Hitung IoU (Jaccard Index)
    iou = area_intersection / (area_union + 1e-10)

    # Hitung Precision dan Recall
    tp = area_intersection
    fp = np.sum(prd) - tp  # False Positive
    fn = np.sum(gt) - tp   # False Negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Hitung F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Hitung J & F (Jaccard + F1) / 2
    j_and_f = (iou + f1_score) / 2
    
    return iou, precision, recall, f1_score, j_and_f

def compute_auc(ious_subset):
    clicks = np.arange(1, len(ious_subset) + 1)
    auc = np.trapezoid(ious_subset, clicks)
    normalized_auc = auc / (clicks[-1] - clicks[0]) if len(clicks) > 1 else auc
    return auc, normalized_auc

def calc_auc(clicks, ious):
    auc = np.trapezoid(ious, clicks)
    normalized_auc = auc / (clicks[-1] - clicks[0]) if len(clicks) > 1 else auc
    return auc, normalized_auc

def calc_auc_all_scenario(results):
    # Skenario klik
    scenario_indices = [
        [0, 1],
        [0, 1, 2],
        [0, 1, 3],
        [0, 1, 3, 4],
        [0, 1, 3, 4, 5],
        [0, 1, 3, 6],
        [0, 1, 3, 6, 7],
        [0, 1, 3, 6, 7, 8],
        [0, 1, 3, 6, 7, 8, 9]
    ]

    # Proses semua hasil
    for result in results:
        ious = result.get('iou', [])
        auc_list = []

        for indices in scenario_indices:
            try:
                ious_subset = [ious[i] for i in indices]
                auc, normalized_auc = compute_auc(ious_subset)
                auc_list.append(normalized_auc)
            except IndexError:
                auc_list.append(None)

        result['auc'] = auc_list

    auc_matrix = []

    for result in results:
        auc_values = result.get('auc', [])
        if len(auc_values) == 9:
            auc_matrix.append(auc_values)

    auc_matrix = np.array(auc_matrix)

    average_aucs = np.nanmean(auc_matrix, axis=0)

    return average_aucs