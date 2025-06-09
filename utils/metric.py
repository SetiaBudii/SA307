import numpy as np
import torch

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