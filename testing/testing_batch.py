import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '..')
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import * 
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.metric import calc_iou
from PIL import Image
from samaug.randomsampling import get_random_point
from npc.neg_prompt_calibration import neg_prompt_calibration

def main(args):
    config = load_config()
    _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    checkpoint = torch.load(args.checkpoint_path,weights_only=False)
    model_state_dict = checkpoint['model_state']
    predictor.model.load_state_dict(model_state_dict, strict=False)
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_mask"])
    output_folder_test = "/kaggle/working/predict_out"  # ganti sesuai kebutuhan
    os.makedirs(output_folder_test, exist_ok=True)
    output_folder = "/kaggle/working/visual_out"  # ganti sesuai kebutuhan
    os.makedirs(output_folder, exist_ok=True)
    # For dataset malay
    # my_images_test_dir = "/kaggle/input/loveda-307/test_my/images_png"
    # my_masks_test_dir = "/kaggle/input/loveda-307/test_my/all_masks_png"
    # test_data = []
    # for image_name in os.listdir("/kaggle/input/loveda-307/test_my/images_png"):
    #     base_name = os.path.splitext(image_name)[0]
    #     test_data.append({
    #         "image": os.path.join(my_images_test_dir, f'{base_name}.jpg'),
    #         "annotation": os.path.join(my_masks_test_dir, f'{base_name}.png')
    #     })
    # print(test_data)

    #define jumlah run
    num_runs = 1
    miou_scores = []

    for run in range(num_runs):
        print(f"\n🔁 Run {run + 1}/{num_runs}")
        ious = []
        initial_iou = 0
        initial_mask = None
        pbar = tqdm(test_data, desc=f"Testing")
        
        for (i, data) in enumerate(pbar):
            img, gt, points, labels = prep_point_image_test(data)
            if len(points) == 0:
                continue
    
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
    
            initial_mask = masks[0]
            initial_iou = calc_iou(masks[0], gt)
    
            #Jika terdapat penambahan titik positif atau negatif
            if args.positive_point > 0 and args.negative_point >= 0:
                point_prompt_aug = []
                for i in range(args.positive_point):
                    point_prompt_aug.append(get_random_point(masks[0]))
                    
                new_prompt = np.concatenate([points, point_prompt_aug], axis=0)
                input_label = np.ones(len(new_prompt), dtype=int)
                
                neg_points, neg_labels = neg_prompt_calibration(
                        masks,
                        img,
                    )
                
                if len(neg_points) > 0 and args.negative_point > 0:
                    neg_points = np.array(neg_points)
                    num_neg = min(args.negative_point, len(neg_points))  # tidak lebih dari jumlah yang tersedia
                    indices = np.random.choice(len(neg_points), size=num_neg, replace=False)
                    sampled_neg_points = neg_points[indices]
                    neg_points_formatted = np.array([[pt[1], pt[0]] for pt in sampled_neg_points])
                    neg_labels = np.zeros(len(neg_points_formatted), dtype=int)
                    new_prompt = np.concatenate([new_prompt, neg_points_formatted], axis=0)
                    input_label = np.concatenate([input_label, neg_labels], axis=0)
                    
                # Result
                masks, scores, logits = predictor.predict(
                    point_coords=new_prompt,
                    point_labels=input_label,
                    multimask_output=True,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]
            elif args.positive_point < 0 and args.negative_point < 0:
                print("Positive point and negative point cannot be less than 0")
                break
    
            iou = calc_iou(masks[0], gt)
            ious.append(iou)

            #save predicted mask menjadi image
            predicted_mask = masks[0].cpu().numpy()
            predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255
            predicted_mask = Image.fromarray(predicted_mask)
            predicted_mask.save(os.path.join(output_folder_test, f"predicted_mask_{i}.png"))

            #plotting image, ground truth, first mask dan last mask
            fig, axes = plt.subplots(1, 4, figsize=(15, 5))
            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[0].axis("off")
            
            axes[1].imshow(gt.squeeze(0).cpu().numpy())
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")
            
            axes[2].imshow(initial_mask)
            axes[2].set_title(f"Prediksi awal\nIoU: {initial_iou:.4f}")
            axes[2].axis("off")
    
            axes[3].imshow(masks[0])
            axes[3].set_title(f"Prediksi akhir\nIoU: {iou:.4f}")
            axes[3].axis("off")
            
            # Simpan ke folder
            filename = f"plot_{i}.png"
            save_path = os.path.join(output_folder, f"result_{filename}")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            pbar.set_postfix({
                "IoU awal": initial_iou,
                "IoU akhir": iou
            })
    
        miou = sum(ious) / len(ious) if ious else 0
        print(f"\n✅ Mean IoU on test set: {miou:.4f}")
        miou_scores.append(miou)
        
    mean_miou = np.mean(miou_scores)
    std_miou = np.std(miou_scores)
    print(f"\n📊 Final Results after {num_runs} runs:")
    print(f"Mean of Mean IoUs: {mean_miou:.4f}")
    print(f"Standard Deviation: {std_miou:.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for SAM2 model")
    parser.add_argument("--positive_point", type=int, required=True, help="jumlah penambahan titik positif")
    parser.add_argument("--negative_point", type=int, required=True, help="jumlah penambahan titik negatif")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    args = parser.parse_args()

    main(args)