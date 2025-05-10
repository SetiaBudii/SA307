import numpy as np
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from scipy.ndimage import convolve
from scipy.ndimage import binary_fill_holes
from skimage.morphology import remove_small_objects, disk


def waterbodies_extraction(image):
    """
    Fungsi untuk mengekstrak objek air dari citra menggunakan metode entropi dan thresholding Otsu.
    Proses : 1. Hitung entropi citra 
             2. Terapkan thresholding Otsu
             3. Lakukan pengisian lubang pada citra biner 
             4. Hapus objek kecil yang tidak signifikan.
             5. Refinement mask untuk menghilangkan area yang tidak diinginkan
             6. Hapus kembali objek kecil yang tidak signifikan.

    Args:
        image_path (str): Path ke citra yang akan diproses.
    Returns:
        dense_binary_img (ndarray): Citra biner hasil ekstraksi objek air.

    Example:
        >>> mask_water = waterbodies_extraction("path/to/image.jpg")
    """


    # img = io.imread(image_path)
    img = image
    entropy_img = entropy(img[:, :, 0], footprint=disk(3))
    thresh = threshold_otsu(entropy_img)
    binary_img = entropy_img <= thresh

    dense_binary_img = binary_fill_holes(binary_img)
    dense_binary_img = remove_small_objects(dense_binary_img, min_size=500)
    dense_binary_img = refine_mask(dense_binary_img, iterations=3, min_neighbors=8)
    dense_binary_img = remove_small_objects(dense_binary_img, min_size=500)

    return dense_binary_img

def refine_mask(binary_img, iterations=1, min_neighbors=6):
    """
    Fungsi untuk memperhalus citra biner dengan menghitung jumlah tetangga.
    Proses : 1. Hitung jumlah tetangga untuk setiap piksel
             2. Pertahankan piksel yang memiliki jumlah tetangga lebih dari ambang batas.
    Args:
        binary_img (ndarray): Citra biner yang akan diperhalus.
        iterations (int): Jumlah iterasi untuk memperhalus citra.
        min_neighbors (int): Ambang batas jumlah tetangga untuk mempertahankan piksel.
    Returns:
        result (ndarray): Citra biner yang telah diperhalus.
    """

    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    result = binary_img.copy()
    for _ in range(iterations):
        neighbor_count = convolve(result.astype(np.uint8), kernel, mode='constant', cval=0)
        result = result & (neighbor_count >= min_neighbors)
    return result



