import numpy as np
import cv2
import os
import torch
from skimage.segmentation import find_boundaries
from tqdm import tqdm

def read_single(data): # read random image and single mask from  the dataset
        ent  = data[np.random.randint(len(data))] # choose random entry
        Img = cv2.imread(ent["image"])[...,::-1]  # read image
        ann_map = cv2.imread(ent["annotation"]) # read annotation

   # merge vessels and materials annotations
        mat_map = ann_map[:,:,0] # material annotation map
        ves_map = ann_map[:,:,2] # vessel  annotaion map
        mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps

   # Get binary masks and points
        inds = np.unique(mat_map)[1:] # load all indices
        if inds.__len__()>0:
              ind = inds[np.random.randint(inds.__len__())]  # pick single segment
        else:
              return read_single(data)

        #for ind in inds:
        mask=(mat_map == ind).astype(np.uint8) # make binary mask corresponding to index ind
        coords = np.argwhere(mask > 0) # get all coordinates in mask
        yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
        return Img,mask,[[yx[1], yx[0]]]

def read_batch(data,batch_size=4):
      limage = []
      lmask = []
      linput_point = []
      for i in range(batch_size):
              image,mask,input_point = read_single(data)
              limage.append(image)
              lmask.append(mask)
              linput_point.append(input_point)

      return limage, np.array(lmask), np.array(linput_point),  np.ones([batch_size,1])

def prepare_data_train(images_path, annotations_path):
    data = []
    for name in os.listdir(images_path):
        if name.endswith(".png"):
            image_path = os.path.join(images_path, name)
            annotation_path = os.path.join(annotations_path, name)

            # Check if both image and mask exist
            if os.path.exists(image_path) and os.path.exists(annotation_path):
                data.append({"image": image_path, "annotation": annotation_path})
            else:
                print(f"Warning: Missing mask for image '{name}' or invalid paths.")
    return data

def prepare_model( model_cfg, model_checkpoint, device="cuda"):
    model = build_sam2(model_cfg, model_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return predictor

def create_optimizer_and_scaler(model, lr=1e-5, weight_decay=4e-5):
    """
    Membuat optimizer dan scaler untuk training dengan mixed precision.

    Args:
        model (torch.nn.Module): Model yang akan dioptimasi.
        lr (float): Learning rate.
        weight_decay (float): Weight decay untuk regularisasi.

    Returns:
        optimizer: Optimizer AdamW.
        scaler: GradScaler untuk mixed precision training.
    """
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    return optimizer, scaler
