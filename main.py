import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os



working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f'{working_dir}/trained_model/fashion_mnist_model.h5'
model = tf.keras.models.load_model(model_path)

classes_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']





def preprocess_image(image):
    img = image.resize((28, 28))
    img = img.convert('L')
    img = np.array(img)/225.0
    img = img.reshape((1,28, 28, 1))
    return img



st.title=('Fashion Mnist APP')
upload_image = st.file_uploader('Choose an image', type=['png', 'jpg', 'jpeg'])
if upload_image is not None:
    image = Image.open(upload_image)
    col1,col2 = st.columns(2)
    with col1:
        resized_image = image.resize((100, 100))
        st.image(resized_image)
    with col2:
        if st.button('Predict'):
            img_array = preprocess_image(image)
            result = model.predict(img_array)
            label = np.argmax(result)
            label_name = classes_names[label]
            st.success(f"your predicted image is {label_name}")




