import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io

# Load the saved model
model = load_model('./cnn_cat_vs_dog.h5')


def preprocess_image(img):
    try:
        img = img.resize((150, 150))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array

    except Exception as e:
        st.error(f"Error Processing Image: {e}")
        return None


# Function to make prediction
def predict(img_array):
    prediction = model.predict(img_array)
    return prediction[0][0]  # Assuming binary classification


# Streamlit app
st.title('Cat or Dog Classifier')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    preprocessed_img = preprocess_image(image)

    if preprocessed_img is not None:
        # Make prediction
        prediction = predict(preprocessed_img)

        # Display result
        st.write("Prediction:")
        if prediction > 0.5:
            st.write("It's a Dog!")
        else:
            st.write("It's a Cat!")

        # Display confidence
        confidence = prediction if prediction > 0.5 else 1 - prediction
        st.write(f"Confidence: {confidence:.2f}")
    else:
        st.error("Failed to process the image. Please try another one.")

st.write("Note: This model classifies images as either cats or dogs.")