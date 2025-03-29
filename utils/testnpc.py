import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision import transforms
# from SAM2 import sam_model_registry
from sam2 import load_model

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_embeddings = None
        self.setup()

    def setup(self):
        self.model = load_model(
                        variant="large",
                        ckpt_path="sam2_hiera_large.pt",
                        device="cpu"
                    )

    def forward(self, images, prompts):
        image_embeddings = self.model.image_encoder(images)
        pred_masks, _, _ = self.decode(images.shape[-2:], prompts, image_embeddings)
        return pred_masks

    def decode(self, image_shape, prompts, image_embeddings):
        pred_masks = []
        for prompt, embedding in zip(prompts, image_embeddings):
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=None, boxes=prompt, masks=None)
            low_res_masks, _ = self.model.mask_decoder(
                image_embeddings=embedding.unsqueeze(0),
                image_pe=self.model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            masks = F.interpolate(low_res_masks, image_shape, mode="bilinear", align_corners=False)
            pred_masks.append(masks.squeeze(1))
        return pred_masks, None, None


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


def main():
    model = Model()
    model.eval()
    image = preprocess_image("test_image.jpg")
    prompt = torch.tensor([[50, 50, 200, 200]])  # Contoh bounding box

    with torch.no_grad():
        mask = model(image, [prompt])

    mask = mask[0].squeeze().cpu().numpy()
    cv2.imwrite("predicted_mask.png", (mask * 255).astype(np.uint8))
    print("Prediksi selesai, hasil disimpan di predicted_mask.png")


if __name__ == "__main__":
    main()
