import cv2
import torch

def combine_masks(mask1_path, mask2_path):
    """
    Menggabungkan dua gambar masking hitam putih menjadi satu tensor.
    
    :param mask1_path: Path ke gambar masking pertama
    :param mask2_path: Path ke gambar masking kedua
    :return: Tensor hasil penggabungan dengan shape (2, H, W)
    """
    # Baca gambar dalam mode grayscale
    mask1 = cv2.imread(mask1_path, cv2.IMREAD_GRAYSCALE)
    mask2 = cv2.imread(mask2_path, cv2.IMREAD_GRAYSCALE)

    # Pastikan ukuran kedua mask sama
    if mask1.shape != mask2.shape:
        raise ValueError("Ukuran kedua mask harus sama!")

    # Konversi ke tensor PyTorch
    tensor1 = torch.tensor(mask1, dtype=torch.float32)  # Format float untuk kompatibilitas DL
    tensor2 = torch.tensor(mask2, dtype=torch.float32)

    # Gabungkan menjadi satu tensor (dimensi baru di axis=0)
    combined_tensor = torch.stack([tensor1, tensor2], dim=0)

    return combined_tensor

# Contoh penggunaan
mask1_path = "./split_masks/mask_area_0.png"  # Ganti dengan path mask pertama
mask2_path = "./split_masks/mask_area_1.png"  # Ganti dengan path mask kedua

result_tensor = combine_masks(mask1_path, mask2_path)
print("Tensor hasil gabungan:", result_tensor.shape)
print(result_tensor)
