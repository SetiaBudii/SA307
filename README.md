
# TA-307

APE untuk eksperimen KoTA-307

### Installation
Clone the repository:
   ```bash
   git clone https://github.com/SetiaBudii/SA307.git -b industrial
```

Install requirements:
```bash 
%cd SA307 
```
```bash 
!pip install -r requirements.txt
```

Install SAM 2:
```bash 
%cd SA307/SAM2
```
```bash 
!pip install -e .
```

### Struktur Dataset yang Digunakan
Dataset yang digunakan terdiri dari citra satelit dan mask segmentasi. Folder disusun sebagai berikut:
```text
dataset/
├── train/
│   ├── images_png/
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── masks_png/
│       ├── mask_001.png
│       ├── mask_002.png
│       └── ...
├── val/
│   ├── images_png/
│   └── masks_png/
├── test/
│   ├── images_png/
│   └── masks_png/
```
Contoh data: https://www.kaggle.com/datasets/fardanaljihad/loveda-307

### Fine-Tuning
```bash 
!python /kaggle/working/SA307/fine-tune/fine-tune.py
```
### Testing
```bash 
!python /kaggle/working/SA307/testing/testing_final.py --positive_point 0 --negative_point 0 --checkpoint_path "/kaggle/input/testtt/pytorch/baseplus/1/fine_tune_baseplus_10epoch.pth" --typepositivepoint 1 --typefirstpoint 1 --typenegativepoint 1
```
argparse arguments:
- positive_point: jumlah titik positif yang akan ditambahkan
- negative_point: jumlah titik negatif yang akan ditambahkan
- checkpoint_path: path ke file checkpoint model
- typefirstpoint: tipe titik pertama, 1 untuk center, 2 untuk random
- typepositivepoint: tipe penambahan titik positif, 1 untuk directional, 2 untuk random
- typenegativepoint: tipe penambahan titik negatif, 1 untuk random, 2 untuk center


