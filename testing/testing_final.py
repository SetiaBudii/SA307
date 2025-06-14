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
from samaug.directional import generate_directional_points
# from npc.npc_hsv import neg_prompt_calibration
from npc.npc_307 import npc
from npc.npc_307 import npc_hsv
from utils.image_utils import read_single_center
import pandas as pd

def save_iou_to_excel(iou_results, output_path):
    df = pd.DataFrame(iou_results)
    df.to_excel(output_path, index=False)
    print(f"IoU results saved to {output_path}")

"""
Testing script SAM 2

argparse arguments:
- positive_point: jumlah titik positif yang akan ditambahkan
- negative_point: jumlah titik negatif yang akan ditambahkan
- checkpoint_path: path ke file checkpoint model
- typefirstpoint: tipe titik pertama, 1 untuk center, 2 untuk random
- typepositivepoint: tipe penambahan titik positif, 1 untuk directional, 2 untuk random
- typenegativepoint: tipe penambahan titik negatif, 1 untuk random, 2 untuk center

output:
- Folder predict_out: berisi mask yang diprediksi
- Folder visual_out: berisi visualisasi hasil prediksi
- Folder test_excel: berisi file excel dengan hasil IoU

example usage:
python test_batch_final.py --positive_point 3 --negative_point 2 --checkpoint_path "path/to/checkpoint.pth" --typefirstpoint 1 --typepositivepoint 1 --typenegativepoint 1
"""

def calc_metrics(prd, gt):
    # Pastikan ground truth (gt) adalah numpy array
    if isinstance(gt, torch.Tensor):
        gt = gt.squeeze(0).cpu().numpy()
    
    # Verifikasi bahwa dimensi prediksi dan ground truth sama
    assert prd.shape == gt.shape
    
    # Reshape untuk memastikan formatnya konsisten
    prd = prd.reshape(prd.size).copy()
    gt = gt.reshape(gt.size)
    
    # Hitung area intersection dan union
    area_intersection = np.logical_and(prd, gt).sum()
    area_union = np.logical_or(prd, gt).sum()

    # Hitung IoU (Jaccard Index)
    iou = area_intersection / (area_union + 1e-10)

    # Hitung Precision dan Recall
    tp = area_intersection
    fp = np.sum(prd) - tp  # False Positive
    fn = np.sum(gt) - tp   # False Negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Hitung F1 Score
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Hitung J & F (Jaccard + F1) / 2
    j_and_f = (iou + f1_score) / 2
    
    return iou, precision, recall, f1_score, j_and_f


def main(args):
    iou_results = [] # List to store IoU results for each image

    # Load configuration and model
    config = load_config()

    # load model and predictor
    _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    checkpoint = torch.load(args.checkpoint_path,weights_only=False)
    model_state_dict = checkpoint['model_state']
    predictor.model.load_state_dict(model_state_dict, strict=False)

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
    num_runs = 5
    miou_scores = []
    jandf = []

    #Process Testing
    print("🔍 Starting testing...")
    for run in range(num_runs):
        print(f"\n🔁 Run {run + 1}/{num_runs}")
        ious = []
        j_and_f_scores = []
        initial_iou = 0
        initial_mask = None
        pbar = tqdm(test_data, desc=f"Testing")
        
        for (i, data) in enumerate(pbar):
            if(args.typefirstpoint == 1):
                img, gt, points, labels = read_single_center(data)
            elif(args.typefirstpoint == 2):
                img, gt, points, labels = prep_point_image_test(data)
            else:
                print("Type first point tidak valid, harus 1 atau 2")
                break

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
                if(args.typepositivepoint == 1):
                    new_prompt, input_label = generate_directional_points(points, args.positive_point)
                else:
                    point_prompt_aug = []
                    for i in range(args.positive_point):
                        point_prompt_aug.append(get_random_point(masks[0]))
                        
                    new_prompt = np.concatenate([points, point_prompt_aug], axis=0)
                    input_label = np.ones(len(new_prompt), dtype=int)

                masks, scores, logits = predictor.predict(
                    point_coords=new_prompt,
                    point_labels=input_label,
                    multimask_output=True,
                )
                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                logits = logits[sorted_ind]

                # neg_points, neg_labels = npc_hsv( masks, img , args.negative_point)
                neg_points, neg_labels = npc( data['annotation'], masks[0], 4, args.typenegativepoint)
                
                # Penambahan titik negatif
                if len(neg_points) > 0:
                    new_prompt = np.vstack((new_prompt, neg_points[:args.negative_point]))
                    input_label = np.concatenate((input_label, neg_labels[:args.negative_point]))
                    
                    # Result prediksi akhir
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
            _, precision, recall, f1_score, j_and_f = calc_metrics(masks[0], gt)
            j_and_f_scores.append(j_and_f)
            ious.append(iou)

            if(args.positive_point > 0):
                iou_results.append({
                    'image_file': data['image'],
                    'Titik':  new_prompt,
                    'label': input_label,
                    'iou_initial': initial_iou,
                    'iou_final': iou,
                    'precision' : precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'j&f': j_and_f
                    
                })
            else:
                iou_results.append({
                    'image_file': data['image'],
                    'Titik':  points,
                    'label': labels,
                    'iou_initial': initial_iou,
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
            mask_awal = (initial_mask > 0).astype(np.uint8) * 255
            mask_awal = Image.fromarray(mask_awal)
            mask_awal.save(os.path.join(output_folder_test, f"awal_mask_{image_id}.png"))

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
            filename = f"plot_{image_id}.png"
            save_path = os.path.join(output_folder, f"result_{filename}")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            pbar.set_postfix({
                "IoU awal": initial_iou,
                "IoU akhir": iou
            })
    
        miou = sum(ious) / len(ious) if ious else 0
        mean_j_and_f = np.mean(j_and_f_scores)
        save_iou_to_excel(iou_results, os.path.join(output_folder_excel, f"iou_results_{args.typefirstpoint}_{args.typepositivepoint}_{args.typenegativepoint}_{args.positive_point}_{args.negative_point}.xlsx"))
        print(f"\n✅ Mean IoU on test set: {miou:.4f}")
        miou_scores.append(miou)
        jandf.append(mean_j_and_f)
        
    mean_miou = np.mean(miou_scores)
    metric_jandf = np.mean(jandf)
    
    print(f"\n📊 Final Results after {num_runs} runs:")
    if (args.typepositivepoint == 1 and args.typefirstpoint == 1 and args.positive_point > 0):
        print(f"Mean IoU with Directional Points and Center First Point: {mean_miou:.4f}")
    elif (args.typepositivepoint == 1 and args.typefirstpoint == 2 and args.positive_point > 0):
        print(f"Mean IoU with Directional Points and Random First Point: {mean_miou:.4f}")
    elif (args.typepositivepoint == 2 and args.typefirstpoint == 1 and args.positive_point > 0):
        print(f"Mean IoU with Random Points and Center First Point: {mean_miou:.4f}")
    elif (args.typepositivepoint == 2 and args.typefirstpoint == 2 and args.positive_point > 0):
        print(f"Mean IoU with Random Points and Random First Point: {mean_miou:.4f}")
    else:
        print(f"Mean IoU: {mean_miou:.4f}") 

    print(f"J&F mean : {metric_jandf:.4f}")

    iou_results.append({
        'mean_miou': mean_miou,
        'J&F':metric_jandf,
    })
    save_iou_to_excel(iou_results, os.path.join(output_folder_excel, f"final_results_{args.typefirstpoint}_{args.typepositivepoint}_{args.typenegativepoint}_{args.positive_point}_{args.negative_point}.xlsx"))
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for SAM2 model")
    parser.add_argument("--positive_point", type=int, required=True, help="jumlah penambahan titik positif")
    parser.add_argument("--negative_point", type=int, required=True, help="jumlah penambahan titik negatif")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--typefirstpoint", type=int, default=1, help="Type penambahan titik pertama, 1 untuk center, 2 untuk random")
    parser.add_argument("--typepositivepoint", type=int, help="Type penambahan positif point, 1 untuk directional, 2 untuk random", default=1)
    parser.add_argument("--typenegativepoint", type=int, help="Type penambahan negative point, 1 untuk random, 2 untuk directional/center", default=1)
    args = parser.parse_args()

    main(args)