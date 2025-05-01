import numpy as np
import torch
import cv2
import os
import sys
from tqdm import tqdm
from datetime import datetime
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import *
from utils.config import load_config
from utils.data_loader import load_dataset

sys.path.insert(0, '..')

config = load_config()
sam2_model , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
optimizer, scaler = set_optimizer_and_scaler(predictor)
set_trainable_layers(False,True,True,predictor)

#load dataset
train_dir = config["fine_tune_path"]["train_dir"]
train_mask_dir = config["fine_tune_path"]["train_mask_dir"]
train_data = load_dataset(train_dir, train_mask_dir)

resume = False

AGRICULTURE_GS = 7
EPOCHS = config["fine_tune_params"]["epochs"]
best_iou = 0
best_miou = 0

for epoch in range(EPOCHS):
    pbar = tqdm(train_data, desc=f"Fine-tuning Epoch {epoch+1}")
    
    for i, data in enumerate(pbar):
        with torch.amp.autocast('cuda'):
            img, gt_img, input_points, input_labels = read_single(data, visualize_data=False)
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

            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            predictor.model.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
           
            pbar.set_postfix({
                "Loss": loss.item(),
                "mIoU": mean_iou,
                "IoU": iou.mean().item()
            })

            if iou.mean().item() > best_iou:
                best_iou = iou.mean().item()
                best_point = input_points
                image_name = data["image"]

            if mean_iou > best_miou:
                best_miou = mean_iou

    # Last iteration of one epoch
    save_ckpts(epoch, len(train_data), predictor, optimizer, scaler, mean_iou, loss)
    
    print(f"Epoch: {epoch+1}, Iteration: {len(train_data)}, mIoU: {mean_iou}")

# Last epoch
save_ckpts(epoch, len(train_data), predictor, optimizer, scaler, mean_iou, loss)
