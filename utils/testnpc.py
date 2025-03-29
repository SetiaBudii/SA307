import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import random
from torchvision import transforms
# from SAM2 import sam_model_registry
from sam2 import load_model
from sam2.sam2_image_predictor import SAM2ImagePredictor
import albumentations as A
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_embeddings = None
        self.high_res_features = None
        self.setup()

    def setup(self):
        self.model = load_model(
                        variant="large",
                        ckpt_path="../pretrained_model/sam2_hiera_large.pt",
                        device="cpu"
                    )

    def forward(self, images, prompts, high_res_features):
        _, _, H, W = images.shape#[n, 3, 1024, 1024]

        image_embeddings = self.model.image_encoder(images)
        pred_masks, ious, res_masks = self.decode((H, W), prompts, image_embeddings, high_res_features )
        return image_embeddings, pred_masks, ious, res_masks

    # def encode(self, images):
    #     self.image_embeddings = self.model.image_encoder(images)
    #     return self.image_embeddings 

    def decode(self, image_shape, prompts, image_embeddings, high_res_features):
        _bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        _, vision_feats, _, _ = self.model._prepare_backbone_features(image_embeddings)

        feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size)
                for feat, feat_size in zip(vision_feats[::-1], _bb_feat_sizes[::-1])][::-1]
        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
        image_embeddings = feats[-1]
        # high_res_features = feats[:-1]

        if image_embeddings is None:
            raise ValueError("No image embeddings found")

        pred_masks = []
        ious = []
        res_masks = []

        # print("image_embeddings shape:", image_embeddings.shape)
        # print("Type of prompts:", type(prompts))
        # print("Content of prompts:", prompts)

        # Pastikan prompt memiliki batch dimensi
        for prompt, embedding in zip(prompts, image_embeddings):
            # 1️⃣ Pastikan `prompt` berbentuk (B, N, 2)
            if len(prompt.shape) == 2:
                prompt = prompt.unsqueeze(0)  # Jadi (1, N, 2)

            # 2️⃣ Pastikan `V` berbentuk (1, N, 1)
            V = torch.ones((prompt.shape[0], prompt.shape[1], 1), device=prompt.device)
            # print(prompt.shape[0],prompt.shape[1])

            # 3️⃣ Gabungkan `V` dengan `prompt`
            e = torch.cat([prompt, V], dim=-1)  # Jadi (B, N, 3)

            # print(e)
            # 4️⃣ Panggil `sam_prompt_encoder`
            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=(e[..., :2], e[..., 2]),  # Pisahkan kembali koordinat & label
                boxes=None,
                masks=None,
            )
            # print("Type of embedding:", type(embedding))
            # 5️⃣ Dekode mask
            low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=True,
                high_res_features=high_res_features,
            )

            # 6️⃣ Interpolasi ke resolusi asli
            masks = F.interpolate(
                low_res_masks,
                image_shape,
                mode="bilinear",
                align_corners=False,
            )

            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
            mask_tensor = pred_masks[-1].squeeze().detach().cpu()  # Hapus dimensi tambahan & pindahkan ke CPU jika perlu

            # Opsional: Jika nilai negatif, ubah ke rentang 0-1
            mask_tensor = (mask_tensor - mask_tensor.min()) / (mask_tensor.max() - mask_tensor.min())

            # Tampilkan gambar
            plt.imshow(mask_tensor.numpy(), cmap='gray')
            plt.colorbar()
            plt.title("Predicted Mask")
            plt.show()

        return pred_masks, ious, res_masks
    
def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou

def calc_iou_matrix(mask_list1, mask_list2):
    iou_matrix = torch.zeros((len(mask_list1), len(mask_list2)))
    for i, mask1 in enumerate(mask_list1):
        for j, mask2 in enumerate(mask_list2):
            iou_matrix[i, j] = calculate_iou(mask1, mask2)
    return iou_matrix


def uniform_sampling(mask, num_points):
    # print(f"mask.shape: {mask.shape}, mask.dtype: {mask.dtype}")  # Debugging

    if len(mask.shape) != 2:
        raise ValueError(f"Expected a 2D mask, but got shape {mask.shape}")

    y_coords, x_coords = np.where(mask > 0)
    coords = np.column_stack((x_coords, y_coords))  # (N, 2)

    if len(coords) == 0:
        return np.zeros((num_points, 2), dtype=np.float32)

    sampled_indices = np.random.choice(len(coords), size=num_points, replace=len(coords) < num_points)
    sampled_points = coords[sampled_indices]

    return sampled_points.astype(np.float32)



def get_point_prompts(gt_mask, num_points=1):
    prompts = []
    for mask in [gt_mask]:
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        po_points = uniform_sampling(mask, num_points)
        na_points = uniform_sampling((~mask.astype(bool)).astype(float), num_points)
        po_point_coords = torch.tensor(po_points, device=mask_tensor.device)
        na_point_coords = torch.tensor(na_points, device=mask_tensor.device)
        point_coords = torch.cat((po_point_coords, na_point_coords), dim=1)
        po_point_labels = torch.ones(po_point_coords.shape[:2], dtype=torch.int, device=po_point_coords.device)
        na_point_labels = torch.zeros(na_point_coords.shape[:2], dtype=torch.int, device=na_point_coords.device)
        point_labels = torch.cat((po_point_labels, na_point_labels), dim=1)
        print("label awal: ", point_labels)
        print("chord awal : ", point_coords)
        in_points = (point_coords, point_labels)
        prompts.append(in_points)
        prompts.append(in_points)
    return prompts

def get_prompts(gt_masks):
    prompts = get_point_prompts(gt_masks, 1)
    return prompts

def visualize(image, bboxes, title="Image with Bounding Boxes"):
    """Menampilkan gambar dengan bounding box."""
    fig, ax = plt.subplots(1, figsize=(6, 6))
    ax.imshow(image)

    # Gambar bounding box
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=2, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

    ax.set_title(title)
    plt.show()

def show_image_with_points(mask, points):
    plt.figure(figsize=(6, 6))
    plt.imshow(mask, cmap="gray")  # Tampilkan mask dalam grayscale

    legend_added = {"positive": False, "negative": False}  # Hindari duplikasi legend

    for point_coords, point_labels in points:
        point_coords = point_coords.cpu().numpy()
        point_labels = point_labels.cpu().numpy()


        # print(f"Original point_labels: {point_labels}")  # Debug: cek apakah ada label 0 dan 1
        # print(f"Original point_coords: {point_coords}")

        num_points = point_coords.shape[1] // 2
        point_coords = point_coords.reshape(num_points, 2)
        point_labels = point_labels.reshape(-1) 

        print(f"Reshaped point_labels: {point_labels}")  # Debug setelah reshaping

        # Pisahkan titik positif dan negatif
        positive_points = point_coords[point_labels == 1]
        negative_points = point_coords[point_labels == 0]

        # print(f"Final Positive Points: {positive_points}")  # Debug hasil akhir
        # print(f"Final Negative Points: {negative_points}")

        # Plot titik positif (merah)
        if positive_points.shape[0] > 0:
            label = "Positive Points" if not legend_added["positive"] else ""
            plt.scatter(positive_points[:, 0], positive_points[:, 1], c='red', label=label, s=30)
            legend_added["positive"] = True

        # Plot titik negatif (biru)
        if negative_points.shape[0] > 0:
            label = "Negative Points" if not legend_added["negative"] else ""
            plt.scatter(negative_points[:, 0], negative_points[:, 1], c='blue', label=label, s=30)
            legend_added["negative"] = True

    plt.legend()
    plt.title("Generated Points on Mask")
    plt.show()

# def neg_prompt_calibration(
#     mask_ious,
#     prompts,
# ):
#     '''
#     mask_ious:[mask_nums,mask_nums]
#     '''
#     point_list = []
#     point_labels_list = []
#     num_points = 1
#     for m in range(len(mask_ious)):
            
#         pos_point_coords = prompts[0][0][m][:num_points].unsqueeze(0) 
#         neg_point = prompts[0][0][m][num_points:].unsqueeze(0)  
#         neg_points_list = []
#         neg_points_list.extend(neg_point[0])

#         indices = torch.nonzero(mask_ious[m] > float(0,1)).squeeze(1)

#         if indices.numel() != 0:
#             # neg_points_list = []
#             for indice in indices:
#                 neg_points_list.extend(prompts[0][0][indice][:num_points])
#             neg_points = random.sample(neg_points_list, num_points)
#         else:
#             neg_points =neg_points_list
            
#         neg_point_coords = torch.tensor([p.tolist() for p in neg_points], device=neg_point.device).unsqueeze(0)

#         point_coords = torch.cat((pos_point_coords, neg_point_coords), dim=1) 

#         point_list.append(point_coords)
#         pos_point_labels = torch.ones(pos_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
#         neg_point_labels = torch.zeros(neg_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
#         point_labels = torch.cat((pos_point_labels, neg_point_labels), dim=1)  
#         point_labels_list.append(point_labels)

#     point_ = torch.cat(point_list).squeeze(1)
#     point_labels_ = torch.cat(point_labels_list)
#     new_prompts = [(point_, point_labels_)]
#     return new_prompts

def neg_prompt_calibration(
    mask_ious,
    prompts,
):
    '''
    mask_ious:[mask_nums,mask_nums]
    '''
    point_list = []
    point_labels_list = []
    num_points = 1
    for m in range(len(mask_ious)):
            
        pos_point_coords = prompts[0][0][m][:num_points].unsqueeze(0) 
        neg_point = prompts[0][0][m][num_points:].unsqueeze(0)  
        neg_points_list = []
        neg_points_list.extend(neg_point[0])

        indices = torch.nonzero(mask_ious[m] > float(0.1)).squeeze(1)

        if indices.numel() != 0:
            # neg_points_list = []
            for indice in indices:
                neg_points_list.extend(prompts[0][0][indice][:num_points])
            neg_points = random.sample(neg_points_list, num_points)
        else:
            neg_points =neg_points_list
            
        neg_point_coords = torch.tensor([p.tolist() for p in neg_points], device=neg_point.device).unsqueeze(0)

        point_coords = torch.cat((pos_point_coords, neg_point_coords), dim=1) 

        point_list.append(point_coords)
        pos_point_labels = torch.ones(pos_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        neg_point_labels = torch.zeros(neg_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        point_labels = torch.cat((pos_point_labels, neg_point_labels), dim=1)  
        point_labels_list.append(point_labels)

    point_ = torch.cat(point_list).squeeze(1)
    point_labels_ = torch.cat(point_labels_list)
    new_prompts = [(point_, point_labels_)]
    return new_prompts

def main():
    # Contoh input: Satu titik per objek
    image = cv2.imread("3050.png")  # Sesuaikan dengan path gambar
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Cek shape setelah konversi
    # print("Image Tensor Shape:", image.shape)  # Harus (1, 3, H, W)
   
    gt_mask = cv2.imread("3050_mask.png", cv2.IMREAD_GRAYSCALE)  # Pastikan path benar
    # print(f"gt_mask.shape: {gt_mask.shape}, dtype: {gt_mask.dtype}")

    # gt_mask = torch.tensor(gt_mask, dtype=torch.float32) / 255.0  # Normalisasi ke [0, 1]

    points = np.array([250, 300])  # (x, y) koordinat objek
    labels = np.array([1])  # 1 = positive point (di dalam objek)

    # Konversi ke tensor (jika model memerlukan format tensor)
    point_coords = torch.tensor(points, dtype=torch.float32).unsqueeze(0).unsqueeze(1)  # Tambahkan dimensi ke-2

    point_labels = torch.tensor(labels, dtype=torch.int64).unsqueeze(0)
    points_tuple = (point_coords, point_labels)
    
    sammodel = load_model(
                        variant="large",
                        ckpt_path="../pretrained_model/sam2_hiera_large.pt",
                        device="cpu"
                    )

    predictor = SAM2ImagePredictor(sammodel)
    predictor.set_image(image_rgb)

    high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]

    model = Model()
    prompt = get_point_prompts(gt_mask,1)

    # show_image_with_points(gt_mask, prompt)
    with torch.no_grad():
         _, soft_masks, _, _ = model(image, points_tuple ,high_res_features)   
    
    for i, (soft_mask, gt_mask) in enumerate(zip(soft_masks, gt_mask)):  
        soft_mask = (soft_mask > 0).float()
        mask_ious = calc_iou_matrix(soft_mask, soft_mask)
        indices = torch.arange(mask_ious.size(0))
        mask_ious[indices, indices] = 0.0

        # # Tampilkan mask IOUs
        # print(f"Mask IOUs for index {i}:")
        # print("Shape:", mask_ious.shape)
        # print("Values:\n", mask_ious)

    new_prompts = neg_prompt_calibration(mask_ious, prompt)
    print("chord,label akhir: ",new_prompts)

if __name__ == "__main__":
    main()
