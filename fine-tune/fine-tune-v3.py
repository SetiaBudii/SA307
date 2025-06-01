import numpy as np
import torch
import cv2
import argparse
import os
import sys
sys.path.insert(0, '..')

from tqdm import tqdm
from datetime import datetime
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import *
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.validatev2 import validate_model
from utils.visualize_plotting import *

def main(args):
    config = load_config()
    sam2_model , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    optimizer = torch.optim.AdamW(
        params = predictor.model.parameters(),
        lr = 5e-6,
        weight_decay = 4e-5
    )
    scaler = torch.amp.GradScaler('cuda')
    predictor.model.image_encoder.train(False)
    predictor.model.sam_prompt_encoder.train(True)
    predictor.model.sam_mask_decoder.train(True)

    #load dataset
    train_dir = config["fine_tune_path"]["train_dir"]
    train_mask_dir = config["fine_tune_path"]["train_dir_mask"]
    train_data = load_dataset(train_dir, train_mask_dir)

    #load dataset validation
    val_dir = config["fine_tune_path"]["val_dir"]
    val_mask_dir = config["fine_tune_path"]["val_dir_mask"]
    val_data = load_dataset(val_dir, val_mask_dir)

    resume = False
    current_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(current_dir, "ckpt")
    checkpoint_path = os.path.join(current_dir, checkpoint_dir, "last_checkpoint.pth")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(current_dir, "logs")
    log_file_path = os.path.join(log_dir, f"training_log_{timestamp}{args.positive_point}_{args.negative_point}.txt")
    log_file_val_path = os.path.join(log_dir, f"val_log_{timestamp}{args.positive_point}_{args.negative_point}.txt")
    metric_dir = os.path.join(current_dir, "metric")

    EPOCHS = config["fine_tune_params"]["epochs"]
    mean_iou = 0
    loss_list = []
    best_miou = 0

    for epoch in range(EPOCHS):
        pbar = tqdm(train_data, desc=f"Fine-tuning Epoch {epoch+1}")
        
        for i, data in enumerate(pbar):
            with torch.amp.autocast('cuda'):
                img, gt_img, input_points, input_labels = prep_point_image(data, args.positive_point, args.negative_point)
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
                avg_loss = torch.tensor(loss_list).mean().item()
                mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

                predictor.model.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            
                pbar.set_postfix({
                    "Loss": avg_loss,
                    "mIoU": mean_iou,
                    "IoU": iou.mean().item()
                })

                # if mean_iou > best_miou:
                #     best_miou = mean_iou
                #     checkpoint_path = os.path.join(current_dir, checkpoint_dir, f"best_checkpoint_(epoch{epoch+1}).pth")
                #     save_ckpts(epoch, len(train_data), predictor, optimizer, scaler, mean_iou, loss,checkpoint_path)


        # Last iteration of one epoch
        # if epoch % 5 == 0 :
        checkpoint_path = os.path.join(current_dir, checkpoint_dir, f"fine_tune_{epoch+1}epoch_{args.positive_point}_{args.negative_point}.pth")
        save_ckpts(epoch, len(train_data), predictor, optimizer, scaler, mean_iou, loss,checkpoint_path)
            
        with open(log_file_path, "a") as log:
            log.write(f"Epoch {epoch+1}, mIoU: {mean_iou:.4f}, Loss: {avg_loss:.4f}\n")
            
        print(f"Train --> Epoch: {epoch+1}, Loss: {avg_loss}, mIoU: {mean_iou}")
        validate_model(predictor, val_data, epoch, args.positive_point, args.negative_point, log_file_val_path)
        predictor.model.train()
        
    # Last epoch
    checkpoint_path = os.path.join(current_dir, checkpoint_dir, f"fine_tune_{epoch+1}epoch.pth")
    save_ckpts(epoch, len(train_data), predictor, optimizer, scaler, mean_iou, loss,checkpoint_path)

    #Save visualization
    epochs_train, miou_train, loss_train = read_log_file(log_file_path)
    plot_miou(epochs_train, miou_train, save_path=os.path.join(metric_dir, f"miou_per_epoch_train_{args.positive_point}_{args.negative_point}.png"))
    plot_loss(epochs_train, loss_train, save_path=os.path.join(metric_dir, f"loss_per_epoch_train_{args.positive_point}_{args.negative_point}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for SAM2 model")
    parser.add_argument("--positive_point", type=int, required=True, help="jumlah titik positif")
    parser.add_argument("--negative_point", type=int, required=True, help="jumlah titik negatif")
    args = parser.parse_args()
    main(args)