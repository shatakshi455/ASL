# Gesture Speak Sign-to-Text Converter (GSSTC)

## Overview
GestureSpeak-Sign-to-Text-Converter (GSSTC) is a machine learning-based application designed to recognize Sign Language gestures and convert them into text or speech. The project leverages Mediapipe Hands for real-time hand tracking, extracts 3D hand landmarks, and utilizes a trained classifier for gesture recognition. GSSTC aims to bridge the communication gap for individuals who rely on sign language by providing an intuitive and accessible solution.

## Features
- **Real-Time Gesture Recognition**: Uses Mediapipe Hands to detect and track hand landmarks.
- **Custom Model for Signs**: A specialized classifier trained on sign language gestures.
- **Web-Based Interface**: Built with Streamlit for easy accessibility.
- **Live Text Formation**: Converts recognized signs into meaningful text.
- **Specialized Model**: Differentiates between similar gestures.
- **Modular Design**: Includes dataset creation, model training, and deployment components.

## Project Structure
```
GestureSpeak-Sign-to-Text-Converter/
├──asl_alphabet_test          # images to test the model
├──create_datasets            # extracting data from images
   └──create_dataset.py
   └──create_datasetKV.py
   └──create_datasetMN.py
├──data                       # images used to train the model 
├──datasets                   # extracted data stored in pickle files
   └──dataset_main.pickle
   └──datasetKV.pickle
   └──datasetMN.pickle
├──models                     # trained models
   └──model_main.pickle
   └──model_scalerKV.pickle
   └──model_scalerMN.pickle
├──pages
   └──Know_About_Project.py                     # other pages of the user interface
   └──Check_From_Images.py
   └──ASL_Reference_Image.py
├──tests                      # testing modules
   └──test_depl.py
   └──test_dataset.py
   └──test_model.py
   └──test_train.py
├──train_classifiers          # training the model
   └──train_classifier.py
   └──train_classifierKV.py
   └──train_classifierMN.py
├──Real_Time_Recognition.py                     # application 
└── README.md                 # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/shatakshi455/GestureSpeak-Sign-to-Text-Converter.git
   cd GestureSpeak-Sign-to-Text-Converter
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run Real_Time_Recognition.py
   ```

## Training the Model
1. Generate the dataset using:
   ```bash
   python create_datasets/create_dataset.py
   python create_datasets/create_datasetMN.py
   python create_datasets/create_datasetKV.py
   ```
2. Train the classifier:
   ```bash
   python train_classifiers/train_classifier.py
   python train_classifiers/train_classifierMN.py
   python train_classifiers/train_classifierKV.py
   ```
3. The trained models will be saved for use in the Streamlit app.
4. Testing :
   ```bash
   pytest -v tests/test_dataset.py
   pytest -v tests/test_model.py
   pytest -v tests/test_train.py
   pytest -v tests/test_depl.py
   ```
---
Developed with ❤️ by Shatakshi & Srishti.
