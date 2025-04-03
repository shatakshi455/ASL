import streamlit as st
import os
from PIL import Image

# Define the directory as the current directory
ASL_IMAGES_DIR = "."  # Current directory
REFERENCE_IMAGE = "ASL_Reference.jpg"  # Ensure the filename is correct

st.set_page_config(page_title="Reference Image", page_icon="👌", layout="wide")

st.title("American Sign Language (ASL) Reference Image")

# Custom Sidebar Navigation
st.sidebar.title("📌 Navigation Menu")
 
# Check if the reference image exists in the current directory
image_path = os.path.join(ASL_IMAGES_DIR, REFERENCE_IMAGE)

if not os.path.exists(image_path):
    st.error(f"🚨 Image '{REFERENCE_IMAGE}' not found in the current directory!")
else:
    # Load and display the image
    image = Image.open(image_path)
    st.image(image, caption="ASL Reference", use_container_width=True)
