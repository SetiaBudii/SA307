import numpy as np
import matplotlib.pyplot as plt
from skimage import io
import cv2
import numpy as np
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.filters import threshold_otsu
from skimage.morphology import label
from skimage.measure import label
from scipy.ndimage import binary_fill_holes
from skimage.measure import regionprops

def get_mask(image):
    img = image
    # Mengonversi citra RGB ke HSV menggunakan OpenCV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    # Pisahkan kanal H (Hue), S (Saturation), dan V (Value) dari citra HSV
    hue_channel = img_hsv[:, :, 0]  # Kanal Hue
    saturation_channel = img_hsv[:, :, 1]  # Kanal Saturation
    value_channel = img_hsv[:, :, 2]  # Kanal Value

    hue_threshold_low = 0  
    hue_threshold_high = 60  
    hue_threshold_low2 = 90 
    hue_threshold_high2 = 180  

    # Tentukan rentang untuk kanal Saturation (S)
    saturation_threshold_low = 70
    saturation_threshold_high = 255

    # Tentukan rentang untuk kanal Value (V)
    value_threshold_low = 0 
    value_threshold_high = 255

    # Terapkan thresholding untuk masing-masing kanal
    hue_mask = ((hue_channel >= hue_threshold_low) & (hue_channel <= hue_threshold_high)) | \
        ((hue_channel >= hue_threshold_low2) & (hue_channel <= hue_threshold_high2))
    saturation_mask = (saturation_channel >= saturation_threshold_low) & (saturation_channel <= saturation_threshold_high)
    value_mask = (value_channel >= value_threshold_low) & (value_channel <= value_threshold_high)

    # Gabungkan hasil mask untuk mendapatkan citra dengan rentang hue, saturation, dan value yang diinginkan
    combined_mask = hue_mask & (saturation_mask & value_mask)

    # Gabungkan citra asli dengan mask untuk mendapatkan citra yang telah difilter
    filtered_img = np.zeros_like(img)
    filtered_img[combined_mask] = img[combined_mask]

    entropy_img = entropy(filtered_img[:, :, 2], footprint=disk(3))

    # Terapkan Otsu's thresholding pada citra entropy
    thresh = threshold_otsu(entropy_img)
    binary_img = entropy_img <= thresh

    binary_img = ~binary_img  # Inversi binary image untuk mendapatkan area yang diinginkan

    #erode citra biner untuk mengurangi noise
    binary_img = binary_img.astype(np.uint8)
    binary_img = cv2.erode(binary_img, np.ones((3, 3), np.uint8), iterations=2)

    binary_img = binary_fill_holes(binary_img)  # Mengisi lubang dalam citra biner
    labeled_img = label(binary_img)  # Memberikan label pada objek dalam citra biner

    # # Cek jumlah objek yang terdeteksi
    # num_objects = np.max(labeled_img)
    # print(f"Jumlah objek terdeteksi: {num_objects}")

    # # Menampilkan hasil
    # fig, axes = plt.subplots(1, 5, figsize=(20, 5))

    # # Citra asli
    # axes[0].imshow(img)
    # axes[0].set_title("Original RGB Image")
    # axes[0].axis('off')

    # # Kanal Hue
    # axes[1].imshow(hue_channel, cmap='hsv')
    # axes[1].set_title("Hue Channel")
    # axes[1].axis('off')

    # # Kanal Saturation
    # axes[2].imshow(saturation_channel, cmap='gray')
    # axes[2].set_title("Saturation Channel")
    # axes[2].axis('off')

    # # Kanal Value
    # axes[3].imshow(value_channel, cmap='gray')
    # axes[3].set_title("Value Channel")
    # axes[3].axis('off')

    # # Citra yang telah difilter berdasarkan Hue, Saturation, dan Value
    # axes[4].imshow(filtered_img)
    # axes[4].set_title("Filtered Image (Based on H, S, V Ranges)")
    # axes[4].axis('off')

    # # Menampilkan citra entropy dan binary image
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(img,)
    # axes[0].set_title("Entropy Image")
    # axes[0].axis('off')
    # axes[1].imshow(binary_img, cmap='gray')
    # axes[1].set_title("Binary Image (Otsu's thresholded)")  
    # axes[1].axis('off')

    #cari 3 label terbesar
    props = regionprops(labeled_img)
    largest_regions = sorted(props, key=lambda x: x.area, reverse=True)[:3]

    # # Plot hasil deteksi objek terbesar
    # fig, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(img)
    # for region in largest_regions:
    #     minr, minc, maxr, maxc = region.bbox
    #     rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
    #     ax.add_patch(rect)
    #     ax.text(minc, minr - 10, f'Area: {region.area}', color='red', fontsize=12)

    # plt.tight_layout()
    # plt.show()

    return binary_img
