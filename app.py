
import warnings

# Ignore DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import os
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image


model = tf.keras.models.load_model('DRD_model.h5')

UPLOAD_FOLDER = 'uploads'

# Create upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def predict_new(path):

    img = cv2.imread(path)
    level=["Mild","Moderate","NO_DR","Proliferate_DR","Severe"] 
    RGBImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    RGBImg= cv2.resize(RGBImg,(224,224))
    plt.imshow(RGBImg)
    image = np.array(RGBImg) / 255.0

    predict=model.predict(np.array([image]))
    pred=np.argmax(predict,axis=1)
    print(predict)
    pr=list(predict[0])
    pred_level=level[pr.index(max(pr))]
    if pred_level=="No_DR":
        return "No Diabetis"
    else:
    # print(f"Predicted: {predictions[pred[0]]}")
        return f"Level- {pred_level}"          


def delete_old_files():
    files = os.listdir(UPLOAD_FOLDER)
    for file in files:
        os.remove(os.path.join(UPLOAD_FOLDER, file))


st.markdown(
        """
        <style>
        .reportview-container {
            background: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

st.title("Diabetic Retinopathy Detection")

    # File uploader for image
delete_old_files()

    # File uploader for image
uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png'])

if uploaded_file is not None:
    # Save the uploaded image to the upload folder
    image_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    st.image(uploaded_file, caption='The Image', use_column_width=True)


    # Check button to run the model
    if st.button('STAERT TEST'):
        if uploaded_file is not None:
           
            with st.spinner('Processing...'):
                # Run the deep learning model
                
                # Use the loaded model for predictions
                prediction=predict_new(image_path)
                st.success(f"Prediction: {prediction}")  # Display the prediction