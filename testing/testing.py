import argparse
from tqdm import tqdm
import os
import sys
sys.path.insert(0, '..')
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import * 
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.metric import calc_iou
from samaug.randomsampling import get_random_point


def main(args):
    config = load_config()
    _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_mask"])

    ious = []
    pbar = tqdm(test_data, desc=f"Testing")

    for (i, data) in enumerate(pbar):
        img, gt, points, labels = read_data(data)
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

        #Jika terdapat penambahan titik positif atau negatif
        if args.positive_point > 0 and args.negative_point >= 0:
            point_prompt_aug = []
            for i in range(args.positive_point):
                point_prompt_aug.append(get_random_point(masks[0]))
                
            new_prompt = np.concatenate([points, point_prompt_aug], axis=0)
            input_label = np.ones(len(new_prompt), dtype=int)
            
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

        pbar.set_postfix({
            "IoU": iou
        })

    miou = sum(ious) / len(ious) if ious else 0
    print(f"\n✅ Mean IoU on test set: {miou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for SAM2 model")
    parser.add_argument("--positive_point", type=int, required=True, help="jumlah penambahan titik positif")
    parser.add_argument("--negative_point", type=int, required=True, help="jumlah penambahan titik negatif")
    args = parser.parse_args()

    main(args)