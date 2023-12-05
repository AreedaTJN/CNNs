import streamlit as st
import numpy as np
from PIL import Image
# import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('ggnet.h5')

# Define labels
labels = ['glioma_tumor', 'no_tumor', 'meningioma_tumor', 'pituitary_tumor']

def preprocess_image(image):
    img = cv2.resize(image, (150, 150))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def predict_tumor(image):
    # Preprocess the image
    img = preprocess_image(image)

    # Make prediction
    prediction = model.predict(img)
    predicted_class = labels[np.argmax(prediction)]
    return predicted_class

def main():
    st.title("Brain Tumor Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the selected image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Make prediction
        prediction = predict_tumor(image_np)
        st.success(f"Predicted Class: {prediction}")

if __name__ == "__main__":
    main()

