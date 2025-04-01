import cv2
import numpy as np
import os

def split_mask_areas(mask_path, output_folder="split_masks"):
    """
    Memisahkan setiap area (kontur) dalam mask menjadi gambar tersendiri.
    
    :param mask_path: Path ke gambar mask biner
    :param output_folder: Folder untuk menyimpan hasil split mask
    """
    # Buat folder output jika belum ada
    os.makedirs(output_folder, exist_ok=True)

    # Baca mask dalam mode grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Binarisasi gambar (pastikan hanya 0 dan 255)
    _, binary_mask = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY)

    # Temukan kontur
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Loop untuk setiap area yang ditemukan
    for idx, contour in enumerate(contours):
        # Buat mask kosong untuk setiap area
        area_mask = np.zeros_like(mask)

        # Gambar area pada mask kosong
        cv2.drawContours(area_mask, [contour], -1, 255, thickness=cv2.FILLED)

        # Simpan hasil sebagai gambar baru
        output_path = os.path.join(output_folder, f"mask_area_{idx}.png")
        cv2.imwrite(output_path, area_mask)
        print(f"Saved: {output_path}")

    print("Semua area telah dipisahkan dan disimpan!")

# Contoh penggunaan
mask_image_path = "3050_mask.png"  # Ganti dengan path mask yang benar
split_mask_areas(mask_image_path)
