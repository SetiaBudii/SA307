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
from npc.neg_prompt_calibration import neg_prompt_calibration

def main(args):
    config = load_config()
    _ , predictor = prepare_model_predictor(config["model"]["config"], config["model"]["checkpoint"], device="cuda")
    checkpoint = torch.load(args.checkpoint_path)
    model_state_dict = checkpoint['model_state']
    predictor.model.load_state_dict(model_state_dict, strict=False)
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
            
            neg_points, neg_labels = neg_prompt_calibration(
                    masks[0],
                    img,
                )
            
            # if len(neg_points) > 0:
            #     if args.negative_point == 1 and len(neg_points) > 0:
            #         #get just one point
            #         neg_random_point = neg_points[np.random.randint(len(neg_points))]
            #         neg_points = np.array([[neg_random_point[1], neg_random_point[0]]])
            #         neg_labels = np.zeros(len(neg_points), dtype=int)
            #         new_prompt = np.concatenate([new_prompt, neg_points], axis=0)
            #         input_label = np.concatenate([input_label, neg_labels], axis=0)
            #     elif args.negative_point == 2 and len(neg_points) > 1:
            #         neg_random_points = neg_points[np.random.randint(len(neg_points), size=2)]
            #         neg_points = np.array([[neg_random_points[0][1], neg_random_points[0][0]], [neg_random_points[1][1], neg_random_points[1][0]]])
            #         neg_points = np.array(neg_points)
            #         neg_labels = np.zeros(len(neg_points), dtype=int)
            #         new_prompt = np.concatenate([new_prompt, neg_points], axis=0)
            #         input_label = np.concatenate([input_label, neg_labels], axis=0)
            #     elif args.negative_point == 3 and len(neg_points) > 2:
            #         neg_random_points = neg_points[np.random.randint(len(neg_points), size=3)]
            #         neg_points = np.array([[neg_random_points[0][1], neg_random_points[0][0]], [neg_random_points[1][1], neg_random_points[1][0]], [neg_random_points[2][1], neg_random_points[2][0]]])
            #         neg_points = np.array(neg_points)
            #         neg_labels = np.zeros(len(neg_points), dtype=int)
            #         new_prompt = np.concatenate([new_prompt, neg_points], axis=0)
            #         input_label = np.concatenate([input_label, neg_labels], axis=0)
            #     elif args.negative_point == 2 and len(neg_points) == 1:
            #         neg_random_point = neg_points[np.random.randint(len(neg_points))]
            #         neg_points = np.array([[neg_random_point[1], neg_random_point[0]]])
            #         neg_labels = np.zeros(len(neg_points), dtype=int)
            #         new_prompt = np.concatenate([new_prompt, neg_points], axis=0)
            #         input_label = np.concatenate([input_label, neg_labels], axis=0)
            #     elif args.negative_point == 3 and len(neg_points) == 2:
            #         neg_random_points = neg_points[np.random.randint(len(neg_points), size=2)]
            #         neg_points = np.array([[neg_random_points[0][1], neg_random_points[0][0]], [neg_random_points[1][1], neg_random_points[1][0]]])
            #         neg_points = np.array(neg_points)
            #         neg_labels = np.zeros(len(neg_points), dtype=int)
            #         new_prompt = np.concatenate([new_prompt, neg_points], axis=0)
            #         input_label = np.concatenate([input_label, neg_labels], axis=0)
            #     elif args.negative_point == 3 and len(neg_points) == 1:
            #         neg_random_point = neg_points[np.random.randint(len(neg_points))]
            #         neg_points = np.array([[neg_random_point[1], neg_random_point[0]]])
            #         neg_labels = np.zeros(len(neg_points), dtype=int)
            #         new_prompt = np.concatenate([new_prompt, neg_points], axis=0)
            #         input_label = np.concatenate([input_label, neg_labels], axis=0)

            if len(neg_points) > 0 and args.negative_point > 0:
                num_neg = min(args.negative_point, len(neg_points))  # tidak lebih dari jumlah yang tersedia
                indices = np.random.choice(len(neg_points), size=num_neg, replace=False)
                sampled_neg_points = neg_points[indices]

                # Konversi (x, y) ke (y, x)
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

        pbar.set_postfix({
            "IoU": iou
        })

    miou = sum(ious) / len(ious) if ious else 0
    print(f"\n✅ Mean IoU on test set: {miou:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing script for SAM2 model")
    parser.add_argument("--positive_point", type=int, required=True, help="jumlah penambahan titik positif")
    parser.add_argument("--negative_point", type=int, required=True, help="jumlah penambahan titik negatif")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    args = parser.parse_args()

    main(args)