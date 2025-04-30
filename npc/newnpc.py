import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from scipy.ndimage import convolve
from skimage.morphology import closing, remove_small_objects, disk

def waterbodies_extraction(image_path):
    img = io.imread(image_path)
    entropy_img = entropy(img[:, :, 0], footprint=disk(3))
    thresh = threshold_otsu(entropy_img)
    binary_img = entropy_img <= thresh

    # --- Tahapan post-processing ---
    # Langkah 1: Filter berdasarkan neighbor (padat/tidak bolong)
    dense_binary_img = refine_mask(binary_img, iterations=2, min_neighbors=8)

    # # Langkah 2: Haluskan tepi objek
    # dense_binary_img = closing(dense_binary_img, disk(2))

    # Langkah 3: Buang objek kecil yang tidak signifikan
    dense_binary_img = remove_small_objects(dense_binary_img, min_size=1000)

    return dense_binary_img

def refine_mask(binary_img, iterations=1, min_neighbors=6):
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    result = binary_img.copy()
    for _ in range(iterations):
        neighbor_count = convolve(result.astype(np.uint8), kernel, mode='constant', cval=0)
        result = result & (neighbor_count >= min_neighbors)
    return result



