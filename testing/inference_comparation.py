import argparse
import matplotlib.pyplot as plt
import os
import numpy as n
import sys
sys.path.insert(0, '..')
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import * 
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.metric import calc_iou
from samaug.randomsampling import get_random_point
from npc.neg_prompt_calibration import neg_prompt_calibration
import numpy as np
from PIL import Image

np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=1)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, 
               edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, 
               edgecolor='white', linewidth=0.5, alpha=0.8)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=1.5))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        
        if point_coords is not None and input_labels is not None:
            show_points(point_coords, input_labels, plt.gca(), marker_size=20)  # Ukuran marker lebih kecil
        
        if box_coords is not None:
            show_box(box_coords, plt.gca())

        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        
        plt.axis('off')
        plt.show()

def main(args):
    config = load_config()
    # _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    _ , predictor = prepare_model_predictor("configs/sam2/sam2_hiera_t.yaml","/kaggle/working/SA307/SAM2/sam2/sam2_hiera_t.yaml", device="cuda")
    checkpoint = torch.load(args.checkpoint_path,weights_only=False)
    model_state_dict = checkpoint['model_state']
    predictor.model.load_state_dict(model_state_dict, strict=False)
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_mask"])
    test_data = test_data[args.indexdata]
    ious = []
    f = 0

    img, gt, points, labels = prep_point_image_test(test_data)
    points = np.array([[args.x, args.y]])
    labels = np.array([1])
    print(points)
    print(labels)
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

        mask_image = (masks[0] * 255).astype(np.uint8)  # Skala 0-1 ke 0-255

        # Membuat objek image dari mask
        mask_pil_image = Image.fromarray(mask_image)
        
        # Menyimpan gambar ke file PNG
        mask_pil_image.save("mask_0.png")
        
        # iou1 = calc_iou(masks[0], gt)
        # iou2 = calc_iou(masks[1], gt)
        # iou3 = calc_iou(masks[2], gt)
        iou1 = calc_iou(masks[0], gt)
        
        # # Ambil nilai IoU tertinggi
        # max_iou = max(iou1, iou2, iou3)
        
        # Tambahkan ke list
        ious.append(iou1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis("off")
    
    axes[1].imshow(gt.squeeze(0).cpu().numpy())
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(img)
    # show_mask(masks[0], axes[2], borders=True)
    show_points(points, labels, axes[2], marker_size=20)
    axes[2].set_title("Predicted Mask + Points")
    axes[2].axis("off")
    
    # Simpan ke folder
    output_folder = "/kaggle/working/output/inference/comparation"  # ganti sesuai kebutuhan
    os.makedirs(output_folder, exist_ok=True)
    mask_0 = masks[0]  # Pastikan mask berada dalam format numpy array
    mask_0_bin = np.uint8(mask_0 * 255)
    # Mengonversi array numpy ke objek Image
    mask_image = Image.fromarray(mask_0_bin)
    
    # Menyimpan gambar mask ke file
    output_path_mask = os.path.join(output_folder, 'hasil_pred.png')

    # Menyimpan gambar ke file PNG
    mask_image.save(output_path_mask)
    filename = f"image.png"
    save_path = os.path.join(output_folder, f"result_{filename}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    plt.close()
 
    miou = sum(ious) / len(ious) if ious else 0
    # print(f"\n✅ Mean IoU on test set: {miou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for SAM2 model")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--x", type=int, required=True, help="X coordinate for the point")
    parser.add_argument("--y", type=int, required=True, help="Y coordinate for the point")
    parser.add_argument("--indexdata", type=int, required=True, help="Index of the data to test")
    args = parser.parse_args()

    main(args)