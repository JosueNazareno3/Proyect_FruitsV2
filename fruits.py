import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Cargar modelo
model = load_model("fruits_model.h5")  # asegúrate de subirlo también a Streamlit
CLASS_NAMES = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"]

# Colores para cada fruta (HEX para Streamlit)
fruit_colors = {
    "banana": "#FFFF00",
    "fresa": "#FF0000",
    "kiwi": "#008000",
    "manzana": "#FF0000",
    "naranja": "#FFA500",
    "pina": "#FFFF00",
    "sandia": "#8B0000",
    "uva": "#800080"
}

def predict_image(image: Image.Image):
    """Preprocesar y predecir la fruta en una imagen"""
    image = image.resize((224, 224))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]), dict(zip(CLASS_NAMES, preds.tolist()))

# Interfaz Streamlit
st.title("🍍 Clasificador de Frutas con IA")
st.write("Sube una imagen o toma una foto para reconocer la fruta.")

# Opción 1: Subir imagen
uploaded_file = st.file_uploader("📤 Subir imagen", type=["jpg", "jpeg", "png"])

# Opción 2: Usar cámara (solo funciona en navegador, no en todos los dispositivos)
camera_input = st.camera_input("📷 Tomar foto")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)
elif camera_input is not None:
    image = Image.open(camera_input)

if image is not None:
    st.image(image, caption="📸 Imagen seleccionada", use_column_width=True)

    # Hacer predicción
    class_name, confidence, probabilities = predict_image(image)

    st.subheader(f"✅ Predicción: {class_name.upper()} ({confidence*100:.1f}%)")

    # Mostrar barras de probabilidad
    st.write("### 🔎 Probabilidades por fruta")
    for fruit, prob in probabilities.items():
        st.progress(min(1.0, prob))  # progress espera [0,1]
        st.text(f"{fruit}: {prob*100:.1f}%")
