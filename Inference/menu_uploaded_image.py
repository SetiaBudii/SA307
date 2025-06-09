import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
sys.path.insert(0, '..')
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from sam2.build_sam import build_sam2
from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra
import torch
from menu_automatic import *
import base64
from io import BytesIO
from utils.config import load_config
from utils.fine_tune_utils import *
from npc.npc_307 import npc_hsv
from samaug.randomsampling import get_random_point

def image_to_base64(image):
    # Convert the image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


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

def plot_image_with_points(image, x, y):
    # Convert the image to numpy array for plotting
    img_array = np.array(image)
    
    # Create a figure
    fig, ax = plt.subplots()
    
    # Display the image
    ax.imshow(img_array)
    
    # Plot the points on the image
    ax.plot(x, y, 'go', markersize=2)  # Green circle for each point

    # Hide the axes for a cleaner view
    ax.axis('off')
    
    # Save the plot to a BytesIO object (in-memory image)
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)  # Rewind the buffer to the beginning

    # Convert the buffer to a PIL Image
    img_with_points = Image.open(buf)

    # Return the image
    return img_with_points


def uploaded_image():
    st.title("Inference interaktif Gambar")
    #Step 1: Upload Gambar dan tangkap koordinat
    st.markdown("""
    <div style="border: 1px solid #ccc; padding: 10px; text-align: center; border-radius: 10px; background-color: #f9f9f9;">    
        <h2>Step 1: Unggah Gambar dan Tangkap Koordinat</h2>
        <p>Unggah gambar yang ingin Anda proses, lalu klik pada gambar untuk menangkap koordinat titik.</p>
    </div><br><br>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    name_uploaded_file = uploaded_file.name if uploaded_file else None
    variant_sam = st.sidebar.selectbox("Pilih varian SAM:",
        ("tiny", "small", "base", "large")
    )
    Add_augmentation = st.sidebar.checkbox("Tambahkan Augmentasi Titik", value=True)
    if Add_augmentation:
        penambahan_titik_positif = st.sidebar.number_input("Jumlah titik positif yang ingin ditambahkan:", min_value=1, max_value=10, value=1)
        penambahan_titik_negatif = st.sidebar.number_input("Jumlah titik negatif yang ingin ditambahkan:", min_value=1, max_value=10, value=1)

    if "points" not in st.session_state:
        st.session_state["points"] = []
    if uploaded_file is not None:
        with Image.open(uploaded_file) as img:
        # Store original dimensions
            original_width, original_height = img.size
            width_downscaled = original_width // 2
            height_downscaled = original_height // 2
            img_display = img.copy()
            
            display_width, display_height = img_display.size

            # Create the ImageDraw object for drawing ellipses
            value = streamlit_image_coordinates(img_display, key="pil", width=width_downscaled)
            if value is not None :
                # Calculate the scaling factor
                scale_x = original_width / display_width
                scale_y = original_height / display_height

                # Adjust the coordinates based on the scaling factor
                chord_x = value["x"] * scale_x
                chord_y = value["y"] * scale_y

                # Append the scaled coordinates to the points
                st.session_state["points"].append((chord_x, chord_y))

            chord_x = value["x"] * original_width // display_width
            chord_y = value["y"] * original_height // display_height
            st.markdown("---")
        
            img_with_points = plot_image_with_points(img, chord_x*2, chord_y*2)
            st.markdown("""
            <div style="border: 1px solid #ccc; padding: 10px; text-align: center; border-radius: 10px; background-color: #f9f9f9;">
                <h2>Step 2: Periksa detail inferensi</h2>
                <img src="data:image/png;base64,{}" alt="image" style="max-width: 100%; height: auto;">
                <div style="padding-top: 20px; text-align: center;" >
                    <p>Koordinat yang dipilih: x = {}, y = {}</p>
                    <p>Nama Gambar: {}</p>
                    <p>Varian Model SAM 2: {}</p>
                    <p>Jumlah titik positif tambahan: {}</p>
                    <p>Max Jumlah titik negatif: {}</p>
                </div>
            </div>
            """.format(image_to_base64(img_with_points), chord_x*2, chord_y*2, name_uploaded_file, variant_sam, penambahan_titik_positif, penambahan_titik_negatif), unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("""
            <div style="border: 1px solid #ccc; padding: 10px; text-align: center; border-radius: 10px; background-color: #f9f9f9;">
                <h2>Step 3: Jalankan Inferensi</h2>
                <p>Tekan tombol Jalankan Inferensi menjalankan inferensi pada gambar yang telah diunggah.</p>
                        <pre>
            </pre>
            </div>
            """, unsafe_allow_html=True)
            if st.sidebar.button("Jalankan Inferensi") and st.session_state["points"]:
                    config = load_config()
                    cpkt = f"checkpoint_{variant_sam}"
                    cfg = f"config_{variant_sam}"

                    cp = f"c_{variant_sam}"

                    # load model and predictor
                    _ , predictor = prepare_model_predictor(config["variant_mapping"][cfg], config["variant_mapping"][cpkt], device="cuda")
        
                    checkpoint = torch.load(config["checkpoint_path"][cp], weights_only=False)
                    model_state_dict = checkpoint['model_state']
                    predictor.model.load_state_dict(model_state_dict, strict=False)

                    predictor.set_image(img)
                    st.session_state["run_inference"] = True

                    input_point = np.array([[chord_x*2, chord_y*2]])  # Example points, replace with actual coordinates
                    input_label = np.array([1])  # 1 for foreground, 0 for background
                    masks, scores, logits = predictor.predict(
                        point_coords=input_point,
                        point_labels=input_label,
                        multimask_output=False,
                    )
                    sorted_ind = np.argsort(scores)[::-1]
                    masks = masks[sorted_ind]
                    scores = scores[sorted_ind]
                    logits = logits[sorted_ind]

                    if Add_augmentation:
                        if penambahan_titik_positif > 0 and penambahan_titik_negatif >= 0:
                            point_prompt_aug = []
                            for i in range(penambahan_titik_positif):
                                point_prompt_aug.append(get_random_point(masks[0]))
                                
                            new_prompt = np.concatenate([input_point, point_prompt_aug], axis=0)
                            input_label = np.ones(len(new_prompt), dtype=int)

                            masks, scores, logits = predictor.predict(
                                point_coords=new_prompt,
                                point_labels=input_label,
                                multimask_output=True,
                            )
                            sorted_ind = np.argsort(scores)[::-1]
                            masks = masks[sorted_ind]
                            scores = scores[sorted_ind]
                            logits = logits[sorted_ind]
                            # neg_points, neg_labels = npc_hsv( masks, img , args.negative_point)
                            neg_points, neg_labels = npc_hsv( masks, img, penambahan_titik_negatif)
                            
                            # Penambahan titik negatif
                            if len(neg_points) > 0:
                                new_prompt = np.vstack((new_prompt, neg_points[:penambahan_titik_negatif]))
                                input_label = np.concatenate((input_label, neg_labels[:penambahan_titik_negatif]), axis=0)
                                # Result prediksi akhir
                                masks, scores, logits = predictor.predict(
                                    point_coords=new_prompt,
                                    point_labels=input_label,
                                    multimask_output=True,
                                )
                                sorted_ind = np.argsort(scores)[::-1]
                                masks = masks[sorted_ind]
                                scores = scores[sorted_ind]
                                logits = logits[sorted_ind]

                            input_point = new_prompt
                            input_label = input_label
                    
                    result_image = show_masks(img, masks, scores, point_coords=input_point, input_labels=input_label)
                    colomns = st.columns(3)
                    with colomns[0]:
                        st.image(img, caption="Gambar asli", use_column_width=True)
                    with colomns[1]:
                        st.image(img, caption="Ground Truth", use_column_width=True)
                    with colomns[2]:
                        st.image(masks[0], caption="Hasil Prediksi", use_column_width=True)

                    #divider
                    st.markdown("---")
                    st.image(result_image, caption="Hasil Akhir", use_column_width=True)
            
            # st.write("Captured Coordinates:", value)
            # st.session_state["points"].append((value["x"], value["y"]))


        if uploaded_file is not None:

            image = Image.open(uploaded_file)
            save_path = os.path.join("images", uploaded_file.name)
            if not os.path.exists("images"):
                os.makedirs("images")
            image.save(save_path)
            st.sidebar.success("Gambar berhasil di-upload!")

            #using Columns to display the image and coordinates
            # with st.echo("below"):
            #     value = streamlit_image_coordinates(
            #         save_path,
            #         key="local",
            #     )

            # st.write(value)

        else:
            st.sidebar.write("Belum ada gambar yang di-upload.")
    else:
        st.sidebar.write("Silakan upload gambar untuk memulai.")