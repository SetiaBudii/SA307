import argparse
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '..')
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import * 
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.metric import calc_iou
from samaug.randomsampling import get_random_point
from npc.neg_prompt_calibration import neg_prompt_calibration
from utils.inference_utils import *
import numpy as np
import cv2
import random
from PIL import Image

np.random.seed(3)

def main(args):
    config = load_config()
    _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    checkpoint = torch.load(args.checkpoint_path,weights_only=False)
    model_state_dict = checkpoint['model_state']
    predictor.model.load_state_dict(model_state_dict, strict=False)
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_mask"])

    output_base = "/kaggle/working/output/hasiltest"
    os.makedirs(output_base, exist_ok=True)
    os.makedirs(f"{output_base}/akhir", exist_ok=True)
    os.makedirs(f"{output_base}/satu_titik", exist_ok=True)
    os.makedirs(f"{output_base}/tiga_titik", exist_ok=True)
    os.makedirs(f"{output_base}/penambahan_neg", exist_ok=True)
    os.makedirs(f"{output_base}/penambahan_pos", exist_ok=True)

    ious = []
    miou = 0

    pbar = tqdm(test_data, desc=f"Testing")
        
    for (i, data) in enumerate(pbar):
        threshold_iou = 0
        mask_threshold = None
        mask_after = None
        last_mask = None
        last_iou = 0
        mask_1 = None
        mask_2 = None
        iousatu = 0
        ioudua = 0
        
        img, gt, points, labels = prep_point_image_test(data)
        point_awal = points
        label_awal = labels

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

        #2 Kemungkinan
        # Kasih aja titik negatif
        labels_neg = np.array([1, 0])
        with torch.no_grad():
            predictor.set_image(img)
            
            masks, scores, logits = predictor.predict(
                point_coords=new_points,
                point_labels=labels_neg,
                multimask_output=True,
            )
        
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            mask_1 = masks[0]
            iousatu = calc_iou(mask_1, gt)

        # Kasih aja titik positif
        labels_pos = np.array([1, 1])
        with torch.no_grad():
            predictor.set_image(img)
            
            masks, scores, logits = predictor.predict(
                point_coords=new_points,
                point_labels=labels_pos,
                multimask_output=True,
            )
        
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
            mask_2 = masks[0]
            ioudua = calc_iou(mask_2, gt)

        if(iousatu > ioudua):
            last_iou = iousatu
            last_mask = mask_1
            labels = labels_neg
        else :
            last_iou = ioudua
            last_mask = mask_2
            labels = labels_pos
    
        if(last_iou < threshold_iou):
            last_mask = mask_threshold
            last_iou = threshold_iou
            labels = label_awal
            new_points = point_awal
        
        ious.append(last_iou)
        
        fig, axes = plt.subplots(1, 6, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(gt.squeeze(0).cpu().numpy())
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        axes[2].imshow(img)
        show_mask(mask_threshold, axes[2], borders=True)
        show_points(point_awal, label_awal, axes[2], marker_size=20)
        axes[2].set_title("Segmentasi awal: " + str(threshold_iou))
        axes[2].axis("off")

        axes[3].imshow(img)
        show_mask(mask_1, axes[3], borders=True)
        show_points(new_points, labels_neg, axes[3], marker_size=20)
        axes[3].set_title("Segmentasi awal + negatif point: " + str(iousatu))
        axes[3].axis("off")

        axes[4].imshow(img)
        show_mask(mask_2, axes[4], borders=True)
        show_points(new_points, labels_pos, axes[4], marker_size=20)
        axes[4].set_title("Segmentasi awal + positif point " + str(ioudua))
        axes[4].axis("off")
        
        axes[5].imshow(img)
        show_mask(last_mask, axes[5], borders=True)
        show_points(new_points, labels, axes[5], marker_size=20)
        axes[5].set_title("Segmentasi akhir: " + str(last_iou))
        axes[5].axis("off")
               
        # Folder 'akhir'
        output_mask_path_akhir = os.path.join(output_base, 'akhir', f"segmentasi_akhir_{i}.png")
        plt.imsave(output_mask_path_akhir, mask_threshold, cmap='gray')

        # Folder 'satu_titik'
        output_mask_path_satu_titik = os.path.join(output_base, 'satu_titik', f"segmentasi_satu_titik_{i}.png")
        plt.imsave(output_mask_path_satu_titik, mask_threshold, cmap='gray')

        # Folder 'tiga_titik'
        output_mask_path_tiga_titik = os.path.join(output_base, 'tiga_titik', f"segmentasi_tiga_titik_{i}.png")
        plt.imsave(output_mask_path_tiga_titik, mask_threshold, cmap='gray')
        print(f"Mask threshold disimpan di {output_mask_path_tiga_titik}")

        # Folder 'penambahan_neg'
        output_mask_path_penambahan_neg = os.path.join(output_base, 'penambahan_neg', f"segmentasi_penambahan_neg_{i}.png")
        plt.imsave(output_mask_path_penambahan_neg, mask_threshold, cmap='gray')
        print(f"Mask threshold disimpan di {output_mask_path_penambahan_neg}")

        # Folder 'penambahan_pos'
        output_mask_path_penambahan_pos = os.path.join(output_base, 'penambahan_pos', f"segmentasi_penambahan_pos_{i}.png")
        plt.imsave(output_mask_path_penambahan_pos, mask_threshold, cmap='gray')
        print(f"Mask threshold disimpan di {output_mask_path_penambahan_pos}")

        filename = f"image_{i}.png"
        save_path = os.path.join(output_base, f"result_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    miou = sum(ious) / len(ious) if ious else 0
    print(f"\n✅ Selamat, Mean IoU yang anda dapat pada test set: {miou:.4f}")
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for SAM2 model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    args = parser.parse_args()

    main(args)