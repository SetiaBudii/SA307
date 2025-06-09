import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from sam2 import load_model
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch


def restart_streamlit():
    """ Function to restart the Streamlit app automatically """
    os.system(f"streamlit run {sys.argv[0]}")

# Check and clear Hydra initialization if needed
def clear_hydra_once():
    try:
        # Only clear if Hydra is initialized
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
            initialize_config_module("sam2_configs", version_base="1.2")

    except Exception as e:
        pass  # Hydra may not be initialized yet

# Call the clear_hydra_once function to clear Hydra
clear_hydra_once()



def show_mask(mask, ax, random_color=False, borders=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=1)
    
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=20):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='o', s=marker_size, 
               edgecolor='white', linewidth=0.5, alpha=0.8)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o', s=marker_size, 
               edgecolor='white', linewidth=0.5, alpha=0.8)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=1.5))

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    # Create a copy of the image to return with modifications
    result_image = np.array(image)

    for i, (mask, score) in enumerate(zip(masks, scores)):
        # Create a figure but do not call plt.show() here
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(result_image)  # Display the image
        
        # Show the mask on the image
        show_mask(mask, ax, borders=borders)
        
        # If point coordinates and labels are provided, plot them
        if point_coords is not None and input_labels is not None:
            show_points(point_coords, input_labels, ax, marker_size=20)  # Size the marker smaller
        
        # If box coordinates are provided, draw the box
        if box_coords is not None:
            show_box(box_coords, ax)

        # Title for the current mask
        if len(scores) > 1:
            ax.set_title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        
        # Hide axis for cleaner view
        ax.axis('off')

        # Draw the canvas to update the figure
        fig.canvas.draw()

        # Convert the canvas to a numpy array (RGBA format)
        result_image = np.array(fig.canvas.renderer.buffer_rgba())

        # Close the figure to prevent memory leaks
        plt.close(fig)

    return result_image

def automatic_with_gt():
    st.title("Automatic with GT")
    
    folder_path = "images/default"  # Ganti dengan folder gambar Anda
    gt_path = "ground_truth/default"  # Ganti dengan folder ground truth Anda

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        st.sidebar.info(f"Folder '{folder_path}' dibuat. Silakan tambahkan gambar ke folder ini.")
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    #Select Type of Inference
    inference_type = st.sidebar.selectbox("Pilih tipe inferensi:", ("None Prompt", "M1", "M2", "M3", "M4"))

    variant_sam = st.sidebar.selectbox("Pilih varian SAM:",
        ("tiny", "small", "base_plus", "large")
    )
    if inference_type != "None Prompt":
        jumlah_penambahan_titik = st.sidebar.selectbox(
            "Jumlah penambahan titik Positif dan Negatif:",
            ("Positif 0, Negatif 0", "Positif 1, Negatif 0", "Positif 1, Negatif 1", "Positif 2, Negatif 0",
             "Positif 2, Negatif 1", "Positif 2, Negatif 2", "Positif 3, Negatif 0",
             "Positif 3, Negatif 1", "Positif 3, Negatif 2", "Positif 3, Negatif 3")
        )
    if files:
        selected_file = st.sidebar.selectbox("Pilih gambar:", files)
        image_path = os.path.join(folder_path, selected_file)
        image = Image.open(image_path)

        columns = st.columns(2)
        with columns[0]:
            st.image(image, caption=f"Gambar: {selected_file}", width=256)
        with columns[1]:
            if inference_type != "None Prompt":
                jumlah_penambahan_titik = jumlah_penambahan_titik.split(",")
                jumlah_penambahan_titik = [int(x.split()[1]) for x in jumlah_penambahan_titik]
                st.write(f"Tipe Inferensi: {inference_type}")
                st.write(f"Jumlah Penambahan Titik: Positif {jumlah_penambahan_titik[0]}, Negatif {jumlah_penambahan_titik[1]}")
                st.write("Koordinat yang dipilih: (Y,Y)")
                st.write("Varian SAM: ", variant_sam)
                match inference_type:
                    case "M1":
                        st.write("Inferensi M1: Titik awal random dan tipe penambahan titik positif bersifat random")
                    case "M2":
                        st.write("Inferensi M2: Titik awal random dan tipe penambahan titik positif bersifat directional")
                    case "M3":
                        st.write("Inferensi M3: Titik awal ditengah area pertanian terluas dan tipe penambahan titik positif bersifat random")
                    case "M4":
                        st.write("Inferensi M4: Titik awal ditengah area pertanian terluas dan tipe penambahan titik positif bersifat directional")
            else:
                st.write("Tidak ada koordinat yang dipilih (None Prompt).")
    else:
        st.sidebar.warning("Tidak ada gambar di folder.")
    
    #Create line divider
    st.markdown("---")
    #Button to run inference
    if "run_inference" not in st.session_state:
        st.session_state["run_inference"] = False

    #button in sidebar
    if st.sidebar.button("Jalankan Inferensi") and selected_file:
        model = load_model(
            variant="large",
            ckpt_path="sam2_hiera_large.pt",
            device="cpu"
        )

        predictor = SAM2ImagePredictor(model)
        checkpoint = torch.load("fine_tune_10epoch_2_2.pth", weights_only=False, map_location="cpu")
        model_state_dict = checkpoint['model_state']
        predictor.model.load_state_dict(model_state_dict, strict=False)

        predictor.set_image(image)
        st.session_state["run_inference"] = True
        input_point = np.array([[100, 100],[200, 200]])  # Example points, replace with actual coordinates
        input_label = np.array([1, 0])  # 1 for foreground, 0 for background    
        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        logits = logits[sorted_ind]
        result_image = show_masks(image, masks, scores, point_coords=input_point, input_labels=input_label)
        st.write(f"Hasil Inferensi --> Akurasi {scores[0]:.3f}")
        colomns = st.columns(3)
        with colomns[0]:
            st.image(image, caption="Gambar asli", use_column_width=True)
        with colomns[1]:
            st.image(image, caption="Ground Truth", use_column_width=True)
        with colomns[2]:
            st.image(masks[0], caption="Hasil Prediksi", use_column_width=True)

        #divider
        st.markdown("---")
        st.image(result_image, caption="Hasil Akhir", use_column_width=True)