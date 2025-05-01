import random
import torch

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
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
    cfg,
    mask_ious,
    prompts,
):
    '''
    mask_ious:[mask_nums,mask_nums]
    '''
    point_list = []
    point_labels_list = []
    num_points = cfg.num_points
    for m in range(len(mask_ious)):
            
        pos_point_coords = prompts[0][0][m][:num_points].unsqueeze(0) 
        neg_point = prompts[0][0][m][num_points:].unsqueeze(0)  
        neg_points_list = []
        neg_points_list.extend(neg_point[0])

        indices = torch.nonzero(mask_ious[m] > float(cfg.iou_thr)).squeeze(1)

        if indices.numel() != 0:
            # neg_points_list = []
            for indice in indices:
                neg_points_list.extend(prompts[0][0][indice][:num_points])
            neg_points = random.sample(neg_points_list, num_points)
        else:
            neg_points =neg_points_list
            
        neg_point_coords = torch.tensor([p.tolist() for p in neg_points], device=neg_point.device).unsqueeze(0)

        point_coords = torch.cat((pos_point_coords, neg_point_coords), dim=1) 

        point_list.append(point_coords)
        pos_point_labels = torch.ones(pos_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        neg_point_labels = torch.zeros(neg_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        point_labels = torch.cat((pos_point_labels, neg_point_labels), dim=1)  
        point_labels_list.append(point_labels)

    point_ = torch.cat(point_list).squeeze(1)
    point_labels_ = torch.cat(point_labels_list)
    new_prompts = [(point_, point_labels_)]
    return new_prompts