import os
import sys
sys.path.insert(0, '..')
import argparse
import json
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import find_boundaries
from tqdm import tqdm
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.fine_tune_utils import prepare_model_predictor, prep_point_image_test
from utils.image_utils import read_single_center
from utils.metric import calc_metrics
from PIL import Image
import matplotlib.pyplot as plt

"""
Testing script SAM 2

argparse arguments:
- checkpoint_path: Path to the model checkpoint file.
- threshold: Maximum number of clicks per image (e.g., 20).
- target: Target accuracy threshold in percent for NoC evaluation (e.g., 0.9 for NoC@90).
- typefirstpoint: Type of the first click: 1 for center of the object, 2 for a random point inside the object.

output:
- logs.json: berisi log inference
- result.json: berisi nilai NoC

example usage:
python test_noc.py --checkpoint_path "path/to/checkpoint.pth" --threshold 20 --target 90 --typefirstpoint 1
"""

def generate_next_click(gt_mask, pred_mask, pos_clicks, neg_clicks):
    fp_mask = np.logical_and(pred_mask == 1, gt_mask == 0).astype(np.uint8)
    fn_mask = np.logical_and(pred_mask == 0, gt_mask == 1).astype(np.uint8)

    fp_area = np.sum(fp_mask)
    fn_area = np.sum(fn_mask)

    if fp_area == 0 and fn_area == 0:
        return None

    if fn_area >= fp_area:
        error_type = 'positive'
        error_mask = fn_mask
    else:
        error_type = 'negative'
        error_mask = fp_mask

    boundary = find_boundaries(error_mask, mode='outer')

    dist_map = distance_transform_edt(error_mask & ~boundary)

    y, x = np.unravel_index(np.argmax(dist_map), dist_map.shape)
    return error_type, [x, y]

def simulate_interactive_segmentation(data, predictor, max_clicks=20, iou_threshold=0.9, typefirstpoint=1):
    pos_clicks = []
    neg_clicks = []

    if typefirstpoint == 1:
        img, gt, input_point, input_label = read_single_center(data)
    elif typefirstpoint == 2:
        img, gt, input_point, input_label = prep_point_image_test(data)

    gt_mask = gt.squeeze(0).cpu().numpy()

    predictor.set_image(img)
    
    init_click = input_point[0]
    pos_clicks.append(init_click)

    iou_log = []

    for click_num in range(1, max_clicks + 1):
        input_point = pos_clicks + neg_clicks
        input_label = [1] * len(pos_clicks) + [0] * len(neg_clicks)

        input_point_np = np.array(input_point, dtype=np.float32)
        input_label_np = np.array(input_label, dtype=np.float32)
        masks, scores, logits = predictor.predict(
            point_coords=input_point_np,
            point_labels=input_label_np,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        pred_mask = masks[0]

        # Calculate IoU
        iou, precision, recall, f1_score, j_and_f = calc_metrics(pred_mask, gt)
        iou_log.append(iou)
        if iou >= iou_threshold:
            return click_num, input_point, input_label, iou_log

        # Add next click
        result = generate_next_click(gt_mask, pred_mask, pos_clicks, neg_clicks)
        if result is None:
            return click_num, input_point, input_label, iou_log
        click_type, coord = result
        if click_type == 'positive':
            pos_clicks.append(coord)
        else:
            neg_clicks.append(coord)

    return max_clicks, input_point, input_label, iou_log, img, gt, pred_mask

def convert(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def main(args):
    # load configuration and model
    config = load_config()

    # type model
    model_name = "404"
    if args.checkpoint_path == "/kaggle/input/testtt/pytorch/tiny/1/fine_tune_tiny_10epoch.pth" or args.checkpoint_path == "tiny":
        model_name = "tiny"
        config["model"]["config"] = config["variant_mapping"]["config_tiny"]
        config["model"]["checkpoint"] = config["variant_mapping"]["checkpoint_tiny"]

    elif args.checkpoint_path == "/kaggle/input/testtt/pytorch/small/1/fine_tune_small_10epoch.pth" or args.checkpoint_path == "small":
        model_name = "small"
        config["model"]["config"] = config["variant_mapping"]["config_small"]
        config["model"]["checkpoint"] = config["variant_mapping"]["checkpoint_small"]

    elif args.checkpoint_path == "/kaggle/input/testtt/pytorch/baseplus/1/fine_tune_baseplus_10epoch.pth" or args.checkpoint_path == "baseplus":
        model_name = "baseplus"
        config["model"]["config"] = config["variant_mapping"]["config_base"]
        config["model"]["checkpoint"] = config["variant_mapping"]["checkpoint_base"]

    elif args.checkpoint_path == "/kaggle/input/cpkt_5/pytorch/2p2n/1/fine_tune_10epoch_2_2.pth":
        model_name = "large"

    # load model and predictor
    _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")

    if args.checkpoint_path not in ["tiny", "small", "baseplus", "large"]:
        checkpoint = torch.load(args.checkpoint_path,weights_only=False)
        model_state_dict = checkpoint['model_state']
        predictor.model.load_state_dict(model_state_dict, strict=False)

    # load data test
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_all_mask"])

    # output folder
    output_folder = config["testing"]["result"]
    os.makedirs(output_folder, exist_ok=True)

    pbar = tqdm(test_data, desc=f"Testing NoC@{int(args.target * 100)}")

    click_num_per_image = []
    logs = []
    for (i, data) in enumerate(pbar):
        filename = os.path.basename(data["image"])

        log = {
            "filename": filename,
            "max_clicks": None,
            "input_point": None,
            "input_label": None,
            "iou": None
        }
        
        max_clicks, input_point, input_label, iou, img, gt, pred_mask = simulate_interactive_segmentation(data, predictor, args.threshold, args.target, args.typefirstpoint)
        click_num_per_image.append(max_clicks)

        log["max_clicks"] = max_clicks
        log["input_point"] = input_point
        log["input_label"] = input_label
        log["iou"] = iou

        logs.append(log)

        pbar.set_postfix({
            "Max Clicks": max_clicks
        })

        #split image name
        image_name = os.path.basename(data['image'])
        image_name = os.path.splitext(image_name)[0]
        image_id = image_name.split('.')[0]

        #save predicted mask menjadi image
        predicted_mask = pred_mask
        predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255
        predicted_mask = Image.fromarray(predicted_mask)
        predicted_mask.save(os.path.join(output_folder, f"predicted_mask_{image_id}.png"))

        #plotting image, ground truth, first mask dan last mask
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img)
        axes[0].set_title("Image")
        axes[0].axis("off")
        
        axes[1].imshow(gt.squeeze(0).cpu().numpy())
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")

        axes[2].imshow(pred_mask)
        axes[2].set_title(f"Result\nIoU: {iou:.4f}")
        axes[2].axis("off")
        
        # Simpan ke folder
        filename = f"plot_{image_id}.png"
        save_path = os.path.join(output_folder, f"result_{filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    
    num_twenty = click_num_per_image.count(20)
    percent_twenty = (num_twenty / len(click_num_per_image)) * 100
    noc_90 = sum(click_num_per_image) / len(click_num_per_image)

    summary = {
        "#image": num_twenty,
        "percent": percent_twenty,
        "summary": noc_90
    }
    logs.append(summary)

    # Simpan ke file JSON
    output_path = os.path.join(output_folder)
    with open(f"{output_path}/{model_name}_logs.json", "w") as f:
        json.dump(logs, f, indent=2, default=convert)

    print(f"\n✅ NoC on test set: {noc_90:.4f}")
    print(f"\n✅ #image: {num_twenty}")
    print(f"\n✅ percent: {percent_twenty:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for NoC metric")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=False,
        help="Path to the model checkpoint file."
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=20,
        help="Maximum number of clicks per image (e.g., 20)."
    )

    parser.add_argument(
        "--target",
        type=float,
        default=0.9,
        help="Target accuracy threshold in percent for NoC evaluation (e.g., 90 for NoC@90)."
    )

    parser.add_argument(
        "--typefirstpoint",
        type=int,
        default=1,
        choices=[1, 2],
        help="Type of the first click: 1 for center of the object, 2 for a random point inside the object."
    )
    
    args = parser.parse_args()

    main(args)