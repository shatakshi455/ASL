import streamlit as st
import os
from PIL import Image
import streamlit_extras.switch_page_button as switch
# Define the directory where ASL images are stored
ASL_IMAGES_DIR = "./asl_alphabet_test"  # Change this to your actual path

# Streamlit page setup
st.title("American Sign Language (ASL) Signs Gallery")
# Custom Sidebar Navigation
st.sidebar.title("📌 Navigation Menu")

# Load images from the directory
if not os.path.exists(ASL_IMAGES_DIR):
    st.error(f"Directory '{ASL_IMAGES_DIR}' not found. Please check the path.")
else:
    # Get list of image files
    image_files = sorted([f for f in os.listdir(ASL_IMAGES_DIR) if f.lower().endswith(('png', 'jpg', 'jpeg'))])
    
    if not image_files:
        st.warning("No images found in the directory.")
    else:
        # Display images in a grid format
        cols = st.columns(7)  # Adjust number of columns as needed
        
        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(ASL_IMAGES_DIR, image_file)
            image = Image.open(image_path)
            
            # Display image with its filename as caption
            with cols[idx % 7]:
                st.image(image, caption=image_file.split('_')[0], use_container_width=True)
