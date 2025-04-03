import streamlit as st

# Set Page Configuration
st.set_page_config(page_title="About the App", page_icon="👐", layout="centered")

# Title
st.markdown(
    """
    <style>
        .big-font {
            font-size: 36px !important;
            font-weight: bold;
            text-align: center;
            color: white;
        }
    </style>
    <p class='big-font'>Gesture Speak Sign-to-Text Converter (GSSTC)</p>
    """,
    unsafe_allow_html=True,
)
st.sidebar.title("📌 Navigation Menu")

# Abstract
st.subheader("Abstract")
st.write(
    "Sign language is one of the oldest and most natural forms of communication. "
    "However, since most people do not know sign language and interpreters are scarce, "
    "GSSTC provides a real-time method using machine learning for fingerspelling-based "
    "American Sign Language recognition. The system processes hand gestures through "
    "MediaPipe for landmark detection, followed by classification using a Random Forest model. "
    "This ensures efficient and accurate gesture recognition, bridging the communication gap."
)

# Introduction
st.subheader("Introduction")
st.write(
    "American Sign Language (ASL) is widely used for communication by the Deaf and Mute (D&M) community. "
    "However, a significant barrier exists as most people do not understand ASL. "
    "This project introduces an AI-powered solution to convert sign gestures into text, "
    "making communication seamless and inclusive."
)

# Motivation
st.subheader("Motivation")
st.write(
    "D&M individuals rely on sign language to communicate, but since many people do not understand ASL, "
    "this creates a communication gap. The goal of GSSTC is to develop a vision-based system "
    "that translates ASL into readable text, allowing effective interaction between ASL users "
    "and non-ASL speakers."
)

# Features Section
st.subheader("Features at a Glance")
st.markdown(
    """
    - **Interactive ASL Guide** – Browse a library of ASL signs with images and descriptions.
    - **Real-Time Sign Detection** – Use your webcam to recognize hand signs instantly.
    - **Upload & Check Images** – Upload an image to check sign recognition.
    - **Training Mode** – Test your signing skills and get instant feedback.
    - **Learn & Practice** – Designed for both deaf and mute individuals for communication and for anyone to learn and practice sign language in a fun way.
    """
)

# Model Used
st.subheader("Model Used")
st.write(
    "GSSTC utilizes **MediaPipe Hands** for precise hand tracking and landmark extraction. "
    "These extracted landmarks are then processed using a **Random Forest Classifier**, "
    "a robust machine learning model capable of distinguishing between different sign gestures "
    "based on key features such as angles and distances between hand landmarks."
)

# Developer Credits
st.subheader("Built with Passion")
st.write(
    "Developed and Designed by IITians as a Software Engineering Lab project. "
    "This project is a step towards making ASL more approachable for everyone."
)

# Closing Note
st.markdown("---")
st.markdown(
    "*We're constantly improving this app! Your feedback helps make ASL learning better.*"
)
st.markdown("Developed with ❤️ by Srishti & Shatakshi.")
