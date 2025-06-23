import os
import sys
sys.path.insert(0, '..')
import argparse
import json
import numpy as np
import torch
from tqdm import tqdm
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.fine_tune_utils import prepare_model_predictor, prep_point_image_test
from utils.image_utils import read_single_center
from utils.metric import calc_metrics, calc_auc_all_scenario
from samaug.randomsampling import get_random_point
from samaug.directional import get_horizontal_point
from npc.npc_307 import npc

"""
Testing script SAM 2

argparse arguments:
- checkpoint_path: path ke file checkpoint model
- typefirstpoint: tipe titik pertama, 1 untuk center, 2 untuk random
- typepositivepoint: tipe penambahan titik positif, 1 untuk directional, 2 untuk random
- typenegativepoint: tipe penambahan titik negatif, 1 untuk random, 2 untuk center

output:
- results.json: berisi log inference
- metric_auc.json: berisi nilai metric auc

example usage:
python test_auc.py --checkpoint_path "path/to/checkpoint.pth" --typefirstpoint 1 --typepositivepoint 1 --typenegativepoint 1
"""

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
    checkpoint = torch.load(args.checkpoint_path,weights_only=False)
    model_state_dict = checkpoint['model_state']
    predictor.model.load_state_dict(model_state_dict, strict=False)

    # load data test
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_all_mask"])

    # output folder
    output_folder = config["testing"]["result"]
    os.makedirs(output_folder, exist_ok=True)

    results = []

    scenario = {
        3: (1, 1),
        4: (2, 0),
        5: (2, 1),
        6: (2, 2),
        7: (3, 0),
        8: (3, 1),
        9: (3, 2),
        10: (3, 3)
    }

    pbar = tqdm(test_data, desc="Testing AUC")

    for (i, data) in enumerate(pbar):
        filename = os.path.basename(data["image"])

        log = {
            "filename": filename,
            "init_point": None,
            "ppa": [],
            "npc": [],
            "iou": [],
            "jnf": [],
            "time": []
        }

        if(args.typefirstpoint == 1):
            img, gt, input_point, input_label = read_single_center(data)
        elif(args.typefirstpoint == 2):
            img, gt, input_point, input_label = prep_point_image_test(data)
        else:
            print("Type first point tidak valid, harus 1 atau 2")
            break

        if len(input_point) == 0:
            continue

        log["init_point"] = input_point[0].tolist()

        predictor.set_image(img)

        # Inference skenario 1
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        iou, precision, recall, f1_score, j_and_f = calc_metrics(masks[0], gt)

        # Save IoU skenario 1
        log["iou"].append(iou)
        log["jnf"].append(j_and_f)

        # PPA
        if(args.typepositivepoint == 1):
            new_point = get_horizontal_point(input_point[0], 3)
            log["ppa"] = new_point
        else:
            for i in range(3):
                random_point = get_random_point(masks[0])
                log["ppa"].append(random_point)

        # Inference skenario 2
        input_point = np.concatenate([input_point, log["ppa"][:1]], axis=0)
        input_label = np.ones(len(input_point), dtype=int)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]

        iou, precision, recall, f1_score, j_and_f = calc_metrics(masks[0], gt)

        # Save IoU skenario 2
        log["iou"].append(iou)
        log["jnf"].append(j_and_f)

        # NPC
        neg_points, neg_labels = npc( data['annotation'], masks[0], 4, args.typenegativepoint)
        log["npc"] = neg_points[:3].tolist()

        # Skenario 3-10
        for i in range(3, 11):
            num_pos_point, num_neg_point = scenario[i]

            input_point = [log["init_point"]]
            input_label = [1]

            # Tambahkan PPA
            input_ppa = log["ppa"][:num_pos_point]
            label_ppa = [1] * len(input_ppa)

            input_point += input_ppa
            input_label += label_ppa

            # Tambahkan NPC
            input_npc = log["npc"][:num_neg_point]
            label_npc = [0] * len(input_npc)

            input_point += input_npc
            input_label += label_npc

            masks, scores, logits = predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]
            scores = scores[sorted_ind]
            logits = logits[sorted_ind]

            iou, precision, recall, f1_score, j_and_f = calc_metrics(masks[0], gt)
            log["iou"].append(iou)
            log["jnf"].append(j_and_f)

        results.append(log)

    auc_summary = calc_auc_all_scenario(results)
    for i, avg in enumerate(auc_summary):
        print(f"AUC skenario {i+2}: {avg:.4f}")

    # Simpan ke file JSON
    output_path = os.path.join(output_folder)
    with open(f"{output_path}/{model_name}_result_auc.json", "w") as f:
        json.dump(results, f, indent=2, default=convert)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for AUC metric")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--typefirstpoint", type=int, help="Type penambahan titik pertama, 1 untuk center, 2 untuk random", default=1)
    parser.add_argument("--typepositivepoint", type=int, help="Type penambahan positif point, 1 untuk directional, 2 untuk random", default=1)
    parser.add_argument("--typenegativepoint", type=int, help="Type penambahan negative point, 1 untuk random, 2 untuk center", default=1)
    args = parser.parse_args()

    main(args)