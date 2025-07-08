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
from utils.metric import calc_iou, calc_metrics
from PIL import Image
from utils.excel import save_iou_to_excel

"""
Testing script (Tanpa menggunakan point) SAM 2

argparse arguments:
--checkpoint_path: Path to the model checkpoint
--variant: Model variant to use

output:
- iou_results: List of IoU results for each image
- predicted_mask: Predicted mask saved as an image

example usage:
python test_prompt_none.py --checkpoint_path "path/to/checkpoint.pth" --variant "base"
"""

def main(args):
    iou_results = [] # List to store IoU results for each image
    status = args.checkpoint_path
    config_key = f"config_{args.variant}"
    checkpoint_key = f"checkpoint_{args.variant}"

    # Load configuration and model
    config = load_config()

    # Load model and predictor
    _ , predictor = prepare_model_predictor(config["variant_mapping"][config_key], config["variant_mapping"][checkpoint_key], device="cuda")

    if args.checkpoint_path != "Base":
        checkpoint = torch.load(args.checkpoint_path,weights_only=False)
        model_state_dict = checkpoint['model_state']
        predictor.model.load_state_dict(model_state_dict, strict=False)
        status = "fine-tuned"

    # Load data test
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_all_mask"])

    # Define output folders
    output_folder_test = config["testing"]["predict_out"]
    os.makedirs(output_folder_test, exist_ok=True)
    output_folder = config["testing"]["visual_out"]
    os.makedirs(output_folder, exist_ok=True)
    output_folder_excel = config["testing"]["excel_out"]
    os.makedirs(output_folder_excel, exist_ok=True)

    #Define total run and miou scores
    num_runs = config["testing"]["num_runs"]
    miou_scores = []
    jandf = []

    #Process Testing
    print("🔍 Starting testing...")
    for run in range(num_runs):
        print(f"\n🔁 Run {run + 1}/{num_runs}")
        ious = []
        j_and_f_scores = []
        
        pbar = tqdm(test_data, desc=f"Testing")
        
        for (i, data) in enumerate(pbar):

            img, gt, _, _ = prep_point_image_test(data)
    
            predictor.set_image(img)
            
            masks, scores, logits = predictor.predict(
                point_coords=None,
                point_labels=None,
                multimask_output=True,
            )
    
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]
                
            iou = calc_iou(masks[0], gt)
            _, precision, recall, f1_score, j_and_f = calc_metrics(masks[0], gt)
            j_and_f_scores.append(j_and_f)
            ious.append(iou)

            iou_results.append({
                'image_file': data['image'],
                'iou_final': iou,
                'precision' : precision,
                'recall': recall,
                'f1_score': f1_score,
                'j&f': j_and_f
            })

            #split image name
            image_name = os.path.basename(data['image'])
            image_name = os.path.splitext(image_name)[0]
            image_id = image_name.split('.')[0]

            #save predicted mask menjadi image
            predicted_mask = masks[0]
            predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255
            predicted_mask = Image.fromarray(predicted_mask)
            predicted_mask.save(os.path.join(output_folder_test, f"predicted_mask_{image_id}.png"))

            #plotting image, ground truth, first mask dan last mask
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(img)
            axes[0].set_title("Image")
            axes[0].axis("off")
            
            axes[1].imshow(gt.squeeze(0).cpu().numpy())
            axes[1].set_title("Ground Truth")
            axes[1].axis("off")

            axes[2].imshow(masks[0])
            axes[2].set_title(f"Prediksi akhir\nIoU: {iou:.4f}")
            axes[2].axis("off")

            # Simpan ke folder
            filename = f"plot_{image_id}.png"
            save_path = os.path.join(output_folder, f"result_{filename}")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            pbar.set_postfix({
                "IoU": iou
            })

        miou = sum(ious) / len(ious) if ious else 0
        mean_j_and_f = np.mean(j_and_f_scores)
        save_iou_to_excel(iou_results, os.path.join(output_folder_excel, f"iou_results_nonePrompt_{status}_.xlsx"))
        print(f"\n✅ Mean IoU on test set: {miou:.4f}")
        miou_scores.append(miou)
        jandf.append(mean_j_and_f)
        
    mean_miou = np.mean(miou_scores)
    metric_jandf = np.mean(jandf)

    print(f"Mean IoU: {mean_miou:.4f}") 
    print(f"J&F mean : {metric_jandf:.4f}")

    iou_results.append({
        'mean_miou': mean_miou,
        'J&F':metric_jandf
    })

    save_iou_to_excel(iou_results, os.path.join(output_folder_excel, f"final_results_nonePrompt_{status}_{args.variant}.xlsx"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for SAM2 model none prompt")
    parser.add_argument("--checkpoint_path", type=str, default="Base", help="Path to the checkpoint file")
    parser.add_argument("--variant", type=str, required=True, choices=["tiny", "small", "base", "large"], help="Variant of the model to use (tiny, small, base, large)")
    args = parser.parse_args()

    main(args)