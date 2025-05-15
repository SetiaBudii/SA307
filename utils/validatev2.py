import numpy as np
import torch
import cv2
import os
import sys
sys.path.insert(0, '..')

from tqdm import tqdm
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import *
from utils.visualize_plotting import *

def validate_model(predictor, val_path, epoch=1, number_positive_points=0, number_negative_points=0, log_file_path=None):
    log_file_val_path = log_file_path
    mean_iou = 0
    loss_list = []
    predictor.model.eval()
    pbar = tqdm(val_path, desc=f"validate in {epoch+1}")
    with torch.no_grad():
        for i, data in enumerate(pbar):
            with torch.amp.autocast('cuda'):
                img, gt_img, input_points, input_labels = read_data(data)
                if gt_img.shape[0] == 0 or len(input_points) == 0: 
                    continue
    
                predictor.set_image(img)
                
                ## Prompt encoder
                mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
                    input_points, 
                    input_labels, 
                    box=None, 
                    mask_logits=None, 
                    normalize_coords=True
                )
                
                sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
                    points=(unnorm_coords, labels),
                    boxes=None,
                    masks=None
                )
            
                ## Mask decoder
                high_res_features = [
                    feat_level[-1].unsqueeze(0) 
                    for feat_level in predictor._features["high_res_feats"]
                ]
                low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
                    image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
                    image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=True,
                    repeat_image=False,
                    high_res_features=high_res_features
                )
            
                ## Upscale the masks to the original image resolution
                prd_masks = predictor._transforms.postprocess_masks(
                    low_res_masks, predictor._orig_hw[-1]
                )
            
                ## Loss Calculation
                prd_mask = torch.sigmoid(prd_masks[:, 0])
                
                seg_loss = (-gt_img * torch.log(prd_mask + 0.00001) - (1 - gt_img) * torch.log((1 - prd_mask) + 0.00001)).mean()
            
                inter = (gt_img * (prd_mask > 0.5)).sum(1).sum(1)
                union = gt_img.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
                iou = inter / (union + 1e-7)
                score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
                loss = seg_loss + score_loss * 0.05
                loss_list.append(loss.item())
                mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
                avg_loss = torch.tensor(loss_list).mean().item()
                
                pbar.set_postfix({
                    "Loss": avg_loss,
                    "mIoU": mean_iou,
                })

    with open(log_file_val_path, "a") as log:
        log.write(f"Epoch {epoch+1}, mIoU: {mean_iou:.4f}, Loss: {avg_loss:.4f}\n")
    
    print(f"Validate --> Epoch: {epoch+1}, mIoU: {mean_iou}, Loss: {avg_loss}")