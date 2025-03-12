import os

main_directory = "../../Dataset/LoveDA/original/val/urban"

# Menyimpan hasil dalam list
folder_counts = []

for folder in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder)

    if not os.path.isdir(folder_path) or not folder.isdigit():  # Lewati jika bukan folder
        continue

    # Pastikan nama folder hanya berisi angka sebelum dikonversi ke integer
    if folder.isdigit():
        folder_counts.append((int(folder), len(os.listdir(f"{folder_path}/images_png"))))
    else:
        folder_counts.append((folder, len(f"{folder_path}/images_png")))  # Simpan string jika bukan angka

# Urutkan: angka dulu, lalu string (jika ada)
folder_counts.sort(key=lambda x: (isinstance(x[0], str), x[0]))

# Cetak hasil
for folder, count in folder_counts:
    print(f"{folder}: {count}")

folder_dict = dict(folder_counts)
print(f"Total: {sum(folder_dict.values())}")
