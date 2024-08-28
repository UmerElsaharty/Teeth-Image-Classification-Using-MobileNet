# Teeth-Image-Classification-Using-MobileNet

## Overview

This repository contains a deep learning image classification project using various Convolutional Neural Networks (CNNs). I experimented with three different CNN architectures:

- **MobileNetV2**
- **EfficientNetB0**
- **ResNet50**

### Model Training

During the training process, I encountered some accuracy issues with both the EfficientNetB0 and ResNet50 models. Initially, their accuracy was suboptimal. However, I was able to solve this problem by unfreezing the layers of the base model. After doing so, the accuracy significantly improved, exceeding 99%.

In contrast, while using MobileNetV2, I did not need to unfreeze the base model layers. Despite this, MobileNetV2 achieved an accuracy of approximately 98%. 

### Why MobileNetV2?

- **High Accuracy**: Reached around 98% accuracy without the need to unfreeze base model layers.
- **Efficiency**: MobileNetV2 is a compact model that performs well in terms of speed, making it suitable for local deployments without GPUs.

## Streamlit Application

To showcase the trained model, I created a Streamlit application that allows users to upload images and get classification results. The app offers the following features:

- **Image Upload**: Users can upload one or more images in JPG, JPEG, or PNG format.
- **Image Display**: Uploaded images are displayed in the sidebar for visual reference.
- **Classification**: The app classifies each image using the MobileNetV2 model and shows the predicted class.
- **Summary Table**: Displays a table of all predictions with the option to download it as an Excel file.
- **Download Predictions**: Users can download the prediction results as an Excel file for further analysis.
- **User Interface**: There are 3 images included in the repo for the User Interface of the app before and after uploading an image from each class
- ### How to Run the Application

1. **Install Dependencies**:
   Ensure you have all the required packages installed. You can install them using:
   ```bash
   pip install tensorflow numpy pandas pillow streamlit openpyxl
2. **Save the Application Code** : Save the Streamlit application code to a file named app.py. You can find this file in the repository.
3. **Run the Streamlit App**: Execute the following command in **command prompt** to run the Streamlit app:
   ```bash
   streamlit run app.py
4. **Access the App**: Open your browser and navigate to the URL provided by Streamlit to interact with the app.

5. **Upload Images and View Results**: Use the app interface to upload one or more images, view classification results, and download the predictions as an Excel file.
