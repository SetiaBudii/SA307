import numpy as np
import cv2
import os
import torch
from tqdm import tqdm
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


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


def read_data(data):
    img = cv2.cvtColor(cv2.imread(data["image"]), cv2.COLOR_BGR2RGB)
    gt_img = cv2.imread(data["annotation"], cv2.IMREAD_GRAYSCALE)
    input_points = []
    input_labels = []

    mask = (gt_img == 7).astype(np.float32)
    if np.any(mask == 1):
        indices = np.argwhere(mask==True)
        random_point = indices[np.random.choice(list(range(len(indices))))]
        random_point = [random_point[1], random_point[0]]

        first_point = random_point
        input_points.append(first_point)
    
        # SAMAug
        for i in range(2):
            # Random Sampling
            indices = np.argwhere(mask==True)
            random_point = indices[np.random.choice(list(range(len(indices))))]
            random_point = [random_point[1], random_point[0]]

            input_points.append(random_point)

    
    input_points = np.array(input_points)
    input_labels = np.ones(len(input_points), dtype=int)

    gt_img = torch.from_numpy(gt_img) 
    gt_img = (gt_img == 7).float()
    gt_img = gt_img.unsqueeze(0).cuda()

    return img, gt_img, input_points, input_labels

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

def prepare_model_predictor( model_cfg, model_checkpoint, device="cuda"):
    """
    Setup model dan predictor SAM 2

    Args:
        model_cfg : path ke config model SAM (example:"configs/sam2/sam2_hiera_l.yaml")
        model_checkpoint : path ke checkpoint atau model .pt/pth dari SAM (example:"kaggle/working/SA307sam2_hiera_large.pt")
        device : device yang digunakan (default: "cuda")
    Returns:
        model : Model SAM yang telah dibangun.
        predictor : Instance dari SAM2ImagePredictor.
    """
    
    model = build_sam2(model_cfg, model_checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    return model,predictor

def set_optimizer_and_scaler(predictor, lr=1e-5, weight_decay=4e-5):
    """
    Setup optimizer dan scaler untuk fine-tuning model.

    Args:
        predictor (torch.nn.Module): Model predictor SAM yang akan difine-tune.
        lr (float): Learning rate.
        weight_decay (float): Weight decay untuk regularisasi.

    Returns:
        optimizer: Optimizer AdamW.
        scaler: GradScaler untuk mixed precision fine-tuning.
    """
    optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler
    return optimizer, scaler

def set_trainable_layers(imageEncoder, promptEncoder, maskDecoder, predictor):
    """
    Mengatur trainable layers untuk image encoder, prompt encoder, dan mask decoder.
    
    Args:
        imageEncoder (bool): Menentukan apakah image encoder dalam mode trainable.
        promptEncoder (bool): Menentukan apakah prompt encoder dalam mode trainable.
        maskDecoder (bool): Menentukan apakah mask decoder dalam mode trainable.
        predictor: Instance dari SAM2ImagePredictor yang memiliki model.
    """
    components = {
        "Image Encoder": (predictor.model.image_encoder, imageEncoder),
        "Prompt Encoder": (predictor.model.sam_prompt_encoder, promptEncoder),
        "Mask Decoder": (predictor.model.sam_mask_decoder, maskDecoder),
    }

    # Atur mode trainable dan cetak informasi jika layer trainable
    for name, (component, is_trainable) in components.items():
        component.train(is_trainable)
        if is_trainable:
            print(f"\n{name}:")
            print(component)

    print("\nModel telah diatur ke mode pelatihan.")

     
def save_ckpts(epoch, itr, predictor, optimizer, scaler, mean_iou, loss, cpkt_path):
    torch.save({
        "epoch": epoch,
        "iteration": itr,
        "model_state": predictor.model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "mean_iou": mean_iou,
        "loss": loss.item()
    }, cpkt_path)