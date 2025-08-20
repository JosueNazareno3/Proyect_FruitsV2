import streamlit as st
import av
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# Cargar modelo
model = load_model("fruits_model.h5")
CLASS_NAMES = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"]

def predict_frame(frame):
    image = frame.resize((224, 224))
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx])

# Configuración para WebRTC
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_image()
        label, conf = predict_frame(img)

        # Convertimos el frame a OpenCV para añadir texto
        img_cv = np.array(img)
        cv2 = __import__("cv2")  # importar dinámicamente
        cv2.putText(img_cv, f"{label} ({conf*100:.1f}%)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img_cv, format="bgr24")

st.title("🍌 Clasificador de Frutas en Vivo")
webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
