import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import pandas as pd
from io import BytesIO
import requests
import os
# Load pre-trained model
# MODEL_PATH = "F:/intern/second week/model.h5"  
# model = load_model(MODEL_PATH)


# Define the URL to download the model file

# Use the raw GitHub URL
MODEL_URL = 'model.h5'
# MODEL_PATH = 'F:/intern/second week/model.h5'  # Adjust the path as needed for your deployment environment

# # Ensure the directory exists
# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# # Download the model file if it doesn't exist locally
# if not os.path.exists(MODEL_PATH):
#     response = requests.get(MODEL_URL)
#     with open(MODEL_PATH, 'wb') as file:
#         file.write(response.content)

# Load the model
model = load_model(MODEL_URL)
low_confidence_threshold = 0.98


# Define class names for the model
class_names = np.array(['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT'])  

# Streamlit app title and description
st.title("🦷 Enhanced Dental Diseases Image classification app 🦷")
st.markdown("Upload one or more images from the **same directory** to classify them .")
st.markdown("The app may recognize other images as a teeth disease so the app is not always accurate.")

# Sidebar for displaying images
st.sidebar.title("Uploaded Images")

# Option to upload multiple images
uploaded_files = st.file_uploader("Choose one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    st.write("### Classification Results")
    predictions = []

    for uploaded_file in uploaded_files:
        # Display the uploaded image in the sidebar
        image = Image.open(uploaded_file)
        st.sidebar.image(image, caption=f'Image: {uploaded_file.name}', use_column_width=True)

        # Preprocess the image
        size = (224, 224)  # Adjust size based on your model's input size
        image = ImageOps.fit(image, size, Image.LANCZOS)  # Resize the image

        image = np.array(image) / 255.0  # Normalize to [0,1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        # Prediction
        with st.spinner("Classifying..."):
            prediction = model.predict(image)
            predicted_class_index = np.argmax(prediction)
            predicted_class = class_names[predicted_class_index]
            confidence_scores = prediction[0]
            confidence = confidence_scores[predicted_class_index]
            
            # Check if all confidence scores are low
            if all(score < low_confidence_threshold for score in confidence_scores):
                st.warning(f"The model is not confident about the classification for {uploaded_file.name}")
                st.warning(f"Consider these steps for better results: ")
                st.warning("1:Upload a clear image ")
                st.warning("2:Make sure that the image's format is jpg,png and jpeg")
                
                predicted_class = "Uncertain"
            
            predictions.append((uploaded_file.name, predicted_class))

        # Display the result for each image
        st.success(f"Predicted Class for {uploaded_file.name}: **{predicted_class}**")

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["Image Name", "Predicted Class"])

    # Display a summary table of all predictions
    st.write("### Summary of Predictions")
    st.table(predictions_df)

    # Download button for the DataFrame as an Excel file
    st.write("### Download Predictions")
    towrite = BytesIO()
    predictions_df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button(
        label="Download predictions as Excel",
        data=towrite,
        file_name='predictions.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Developer notes
    st.markdown("---")
    st.header("💻 Developer's Note")
    st.write("Thank you for using this enhanced image classification app! Feel free to experiment with different images.")
else:
    st.write("Please upload one or more image files from the **same directory**.")

# Footer with contact information
st.markdown("---")
st.write("👨‍💻 Developed by Omar El-saharty ")
