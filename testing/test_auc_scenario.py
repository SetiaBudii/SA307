import sys
sys.path.insert(0, '..')
import argparse
from utils.metric import calc_auc

"""
Script for compute the AUC (Area Under Curve) metric for interactive segmentation based on clicks and IoU values

argparse arguments:
- clicks
- ious

example usage:
python test_auc_per_scenario.py --clicks [1, 2, 3, 4] --ious [0.4, 0.45, 0.60, 0.61]
"""

def main(args):
    auc, norm_auc = calc_auc(args.clicks, args.ious)
    print(f"Clicks: {args.clicks}")
    print(f"mIoU: {args.ious}")
    print(f"AUC: {norm_auc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute the AUC (Area Under Curve) metric for interactive segmentation based on clicks and IoU values"
    )

    parser.add_argument(
        "--clicks",
        type=int,
        nargs='+',
        required=True,
        help="List of interaction counts (clicks) at which segmentation accuracy is measured. Example: --clicks 1 2 3 4"
    )

    parser.add_argument(
        "--ious",
        type=float,
        nargs='+',
        required=True,
        help="List of IoU scores corresponding to each number of clicks. Example: --ious 0.45 0.67 0.72 0.78"
    )

    args = parser.parse_args()
    main(args)