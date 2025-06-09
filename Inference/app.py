import streamlit as st
from PIL import Image
import numpy as np
import os
import sys
sys.path.insert(0, '..')
from streamlit_image_coordinates import streamlit_image_coordinates
from menu_automatic import automatic_with_gt
from menu_uploaded_image import uploaded_image

np.random.seed(3)

st.sidebar.markdown("""
# **SAM2 Inference By KoTA 307**
Aplikasi ini digunakan untuk melakukan inferensi pada model SAM2.
""")
st.sidebar.markdown("---")

menu = st.sidebar.radio("Inference Options:", ("Automatic with GT", "Uploaded Image"))

if menu == "Automatic with GT":
    automatic_with_gt()
elif menu == "Uploaded Image":
    uploaded_image()

