import matplotlib.pyplot as plt
import re
import os

def read_log_file(file_path):
    """
    Membaca file log dan mengekstrak nilai epoch, mIoU, dan Loss.

    Args:
        file_path (str): Path ke file log.
    
    Returns:
        tuple: (list epochs, list mIoU, list Loss)
    """
    epochs, miou, loss = [], [], []

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(r"Epoch (\d+), mIoU: ([0-9.]+), Loss: ([0-9.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                miou.append(float(match.group(2)))
                loss.append(float(match.group(3)))
    
    return epochs, miou, loss

def plot_miou(epochs, miou, save_path='miou_per_epoch.png'):
    """
    Membuat dan menyimpan grafik mIoU per epoch.

    Args:
        epochs (list): List epoch.
        miou (list): List mIoU.
        save_path (str): Nama file output.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, miou, marker='o', color='blue')
    plt.title('mIoU per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Grafik mIoU disimpan di: {save_path}")

def plot_loss(epochs, loss, save_path='loss_per_epoch.png'):
    """
    Membuat dan menyimpan grafik Loss per epoch.

    Args:
        epochs (list): List epoch.
        loss (list): List loss.
        save_path (str): Nama file output.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, marker='s', color='red')
    plt.title('Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[INFO] Grafik Loss disimpan di: {save_path}")

# Contoh penggunaan:
# file_path = 'validate_log_2025-05-04_15-46-35.txt'
# epochs, miou, loss = read_log_file(file_path)
# plot_miou(epochs, miou)
# plot_loss(epochs, loss)
