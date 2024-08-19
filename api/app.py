import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the model with st.cache_resource to optimize loading
@st.cache_resource
def load_keras_model():
    return load_model('my_model.keras')

model = load_keras_model()

# Define the Streamlit app
st.title('Plant Disease Prediction')

st.write("Upload an image of a plant leaf:")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    # Prepare the image for model prediction
    img = img.resize((256, 256))  # Adjust size according to your model's input size
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize the image if the model was trained with normalized images
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    # Make prediction
    prediction = model.predict(img_array)
    
    # Process the prediction
    class_names = [
        'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
        'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
        'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
        'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
        'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
        'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
        'Tomato_healthy'
    ]
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Display prediction results
    st.write(f"Predicted Disease: {predicted_class}")
    st.write(f"Confidence: {confidence:.2f}")
