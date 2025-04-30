from tqdm import tqdm
import os
import sys
from torch.onnx.symbolic_opset11 import hstack
from utils.fine_tune_utils import * 
from utils.config import load_config
from utils.data_loader import load_dataset
from utils.metric import calc_iou
sys.path.insert(0, '..')

config = load_config()
sam2_model , predictor = prepare_model_predictor("configs/sam2/sam2_hiera_l.yaml","/kaggle/working/SA307/sam2_hiera_large.pt",device="cuda")
test_data = load_dataset(config["testing"]["test_dir"], config["testing"]["test_dir_mask"])

ious = []
max_iou = 1
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

    iou = calc_iou(masks[0], gt)
    if iou < max_iou:
        max_iou = iou
        max_img = data["image"]
        max_point = points
    ious.append(iou)

    if iou < 0.1:
        print(data["image"])
        print(points)

    pbar.set_postfix({
        "IoU": iou
    })

miou = sum(ious) / len(ious) if ious else 0

print(f"\n✅ Mean IoU on test set: {miou:.4f}")
print(f"📊 Number of valid samples used: {len(ious)} / {len(test_data)}")

print(max_iou)
print(max_img)
print(max_point)