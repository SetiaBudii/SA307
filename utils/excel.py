import pandas as pd

def save_iou_to_excel(iou_results, output_path):
    df = pd.DataFrame(iou_results)
    df.to_excel(output_path, index=False)
    print(f"IoU results saved to {output_path}")