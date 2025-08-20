import numpy as np
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import tflite_runtime.interpreter as tflite
from PIL import Image

# Cargar modelo TFLite
interpreter = tflite.Interpreter(model_path="fruits_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Etiquetas de tus frutas
CLASSES = ["Banana", "Manzana", "Naranja", "Uva", "Piña", "Mango", "Fresa", "Cereza"]

def predict(image):
    img = cv2.resize(image, (100, 100))  # ajusta según tu modelo
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    return CLASSES[np.argmax(preds)], np.max(preds)

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        label, conf = predict(img)
        cv2.putText(img, f"{label} ({conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

st.title("🍌🍎 Clasificador de Frutas en Vivo")
webrtc_streamer(key="fruits", video_transformer_factory=VideoTransformer)
