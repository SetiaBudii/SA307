import argparse
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '..')
from utils.fine_tune_utils import * 
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.metric import calc_iou
from utils.inference_utils import *
import numpy as np
import cv2
import random

np.random.seed(3)

def main(args):
    config = load_config()
    _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    checkpoint = torch.load(args.checkpoint_path,weights_only=False)
    model_state_dict = checkpoint['model_state']
    predictor.model.load_state_dict(model_state_dict, strict=False)
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_mask"])
    ious = []
    miou = 0

    pbar = tqdm(test_data, desc=f"Testing")
        
    for (i, data) in enumerate(pbar):
        threshold_iou = 0
        mask_threshold = None
        iou_after = 0
        mask_after = None
        last_mask = None
        last_iou = 0

        img, gt, points, labels = prep_point_image_test(data)
    
        #Suruh model melakukan segmentasi awal menggunakan titik
        if len(points) == 0:
            return
    
        with torch.no_grad():
            predictor.set_image(img)
            
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
        
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            mask_threshold = masks[0]
            threshold_iou = calc_iou(mask_threshold, gt)
    
        #Suruh model melakukan segmentasi menggunakan 3 titik
        first_point = points[0]  # Titik pertama dari hasil prep_point_image_test
        new_points = [(first_point[0] - 5, first_point[1]),(first_point[0] + 5, first_point[1])] 
        points = np.concatenate([points, new_points], axis=0)
        new_labels = np.ones(len(new_points), dtype=int)  # Semua titik baru adalah titik positif
        labels = np.concatenate([labels, new_labels], axis=0)
    
        with torch.no_grad():
            predictor.set_image(img)
            
            masks, scores, logits = predictor.predict(
                point_coords=points,
                point_labels=labels,
                multimask_output=True,
            )
        
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            mask_after = masks[0]
            iou_after = calc_iou(mask_after, gt)
    
        #Xor kan mask threshold dan mask after untuk menemukan perubahan
        mask_threshold = mask_threshold.astype(np.uint8)
        mask_after = mask_after.astype(np.uint8)
    
        # Menghitung perubahan menggunakan XOR
        mask_perubahan = cv2.bitwise_xor(mask_threshold, mask_after)
    
        #Mencari area perubahan terluas
        contours, _ = cv2.findContours(mask_perubahan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Mencari kontur dengan luas terbesar
        max_area = 0
        max_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > max_area:
                max_area = area
                max_contour = contour
        
        # Mengambil titik acak dari kontur terbesar
        if max_contour is not None and len(max_contour) > 0:
            random_point = random.choice(max_contour.reshape(-1, 2))  # Mengambil koordinat acak
            # Mengembalikan titik acak
            point = random_point
        else:
            print("Tidak ada kontur ditemukan.")
            point = None
    
        # Menambahkan titik hasil terluas ke dalam points
        new_points = np.array([first_point, random_point])
        
        #Bandingkan threshold dengan iou after
    
        if(iou_after < threshold_iou):
            # Jika IoU setelah lebih kecil dari threshold, label menjadi [1, 0]
            labels = np.array([1, 0])  # Label untuk first_point dan random_point
            print("NPC detected")
        else:
            # Jika IoU setelah lebih besar atau sama dengan threshold, label menjadi [1, 1]
            new_points = points
            labels = np.array([1,1,1])  # Label untuk first_point dan random_point
    
        with torch.no_grad():
            predictor.set_image(img)
            
            masks, scores, logits = predictor.predict(
                point_coords=new_points,
                point_labels=labels,
                multimask_output=True,
            )
        
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            
            last_mask = masks[0]
            last_iou = calc_iou(last_mask, gt)
    
        if(last_iou < threshold_iou):
            last_mask = mask_threshold
            last_iou = threshold_iou
        
        ious.append(last_iou)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")
        
        axes[1].imshow(img)
        show_mask(mask_threshold, axes[1], borders=True)
        show_points(new_points, labels, axes[1], marker_size=20)
        axes[1].set_title("Predicted Mask + Points")
        axes[1].axis("off")
        
        axes[2].imshow(img)
        show_mask(last_mask, axes[2], borders=True)
        show_points(new_points, labels, axes[2], marker_size=20)
        axes[2].set_title("Predicted Mask + Points")
        axes[2].axis("off")
        
        # Simpan ke folder
        output_folder = "/kaggle/working/output/testv4"  # ganti sesuai kebutuhan
        os.makedirs(output_folder, exist_ok=True)
        filename = f"image_{i}.png"
        save_path = os.path.join(output_folder, f"result_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    miou = sum(ious) / len(ious) if ious else 0
    print(f"\n✅ Mean IoU on test set: {miou:.4f}")
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for SAM2 model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    args = parser.parse_args()

    main(args)