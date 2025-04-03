import numpy as np
import torch
import cv2
import os
import sys
from tqdm import tqdm
from datetime import datetime
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import *

sys.path.insert(0, '..')

sam2_model , predictor = prepare_model_predictor("configs/sam2/sam2_hiera_l.yaml","/kaggle/working/SA307/sam2_hiera_large.pt",device="cuda")
optimizer, scaler = set_optimizer_and_scaler(predictor)
set_trainable_layers(False,True,True,predictor)

resume = False # Set ke True jika ingin melanjutkan dari checkpoint

# Read ckpt
current_dir = os.path.dirname(os.path.abspath(__file__))
checkpoint_dir = os.path.join(current_dir, "ckpt")
checkpoint_path = os.path.join(current_dir, checkpoint_dir, "last_checkpoint.pth")

# Resume atau fine tune dari awal
if resume and os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    predictor.model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scaler.load_state_dict(checkpoint["scaler_state"])
    mean_iou = checkpoint["mean_iou"]
    start_itr = checkpoint["iteration"] + 1 
    print(f"Resuming training from iteration {start_itr}, mIoU: {mean_iou:.4f}")
elif resume and not os.path.exists(checkpoint_path):
    print("Checkpoint tidak ditemukan, melanjutkan dari awal training...")
else:
    print("Mulai training dari awal...")
    mean_iou = 0 
    start_itr = 0

# Define log file path
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(current_dir, "logs")
log_file_path = os.path.join(log_dir, f"training_log_{timestamp}.txt")

#prepare data
data = prepare_data_train("data/images", "data/annotations")

#cek data
if len(data) == 0:
    print("Tidak ada data untuk dilatih. Pastikan path gambar, anotasi benar, dan path sudah benar.")
    sys.exit(1)

# Training loop
for itr in tqdm(range(start_itr, 20), desc="Training", position=0, leave=True):
    with torch.cuda.amp.autocast():
        image, mask, input_point, input_label = read_batch(data, batch_size=4)
        if len(image) == 0 or mask.shape[0] == 0:
            print(f"Iteration {itr}: Batch is empty, skipping...")
            continue

        image = [img.astype(np.float32) / 255.0 for img in image]
        predictor.set_image_batch(image)

        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(
            input_point, input_label, box=None, mask_logits=None, normalize_coords=True
        )
        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None
        )

        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"],
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )
        prd_masks = predictor._transforms.postprocess_masks(
            low_res_masks, predictor._orig_hw[-1]
        )

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])
        seg_loss = (
            -gt_mask * torch.log(prd_mask + 1e-5)
            - (1 - gt_mask) * torch.log((1 - prd_mask) + 1e-5)
        ).mean()

        inter = (gt_mask * (prd_mask > 0.5)).sum(dim=(1, 2))
        union = gt_mask.sum(dim=(1, 2)) + (prd_mask > 0.5).sum(dim=(1, 2)) - inter
        iou = inter / (union + 1e-5)
        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()

        loss = seg_loss + score_loss * 0.05
        predictor.model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Save model periodically & last checkpoint
        if itr % 999 == 0:
            # Save the model weights
            torch.save(predictor.model.state_dict(), os.path.join(checkpoint_dir, f"model_{itr}.pth"))
            print(f"Saved model at step {itr}")

            # Save checkpoint (including mIoU, loss, and model state)
            torch.save({
                "iteration": itr,
                "model_state": predictor.model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "mean_iou": mean_iou,  # Save mIoU
                "loss": loss.item()  # Save loss
            }, checkpoint_path)

            # Log loss and mIoU to a file
            with open(log_file_path, "a") as log:
                log.write(f"Step {itr}, mIoU: {mean_iou:.4f}, Loss: {loss.item():.4f}\n")

        # Update mean_iou
        mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())
        tqdm.write(f"Step {itr}, Accuracy(IOU)={mean_iou:.4f}")
