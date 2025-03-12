import os
import cv2

# Folder input (ground truth grayscale mask)
image_folder = "../../Dataset/LoveDA/original/train/rural/masks_agriculture"  # Cek path sebelum menjalankan!

# Folder output untuk binary mask
output_folder = "../../Dataset/LoveDA/original/train/rural/binary_agriculture"

# Buat folder output jika belum ada
os.makedirs(output_folder, exist_ok=True)

# Nilai threshold (atur sesuai kebutuhan)
threshold = 1  # Piksel >= 127 menjadi 255, sisanya menjadi 0

# Loop untuk memproses setiap gambar dalam folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    output_path = os.path.join(output_folder, image_name)

    print(f"Processing image: {image_name}")

    # Baca gambar dalam mode grayscale
    ann_map = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Pastikan gambar berhasil dibaca
    if ann_map is None:
        print(f"Warning: Gagal membaca {image_name}, dilewati.")
        continue

    # Konversi ke binary mask dengan thresholding
    _, binary_mask = cv2.threshold(ann_map, threshold, 255, cv2.THRESH_BINARY)

    # Simpan hasil binary mask
    cv2.imwrite(output_path, binary_mask)

print("Proses konversi selesai!")
