from flask import Flask, Response, render_template_string
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Cargar modelo (asegúrate de tener tu modelo .h5 entrenado para estas frutas)
model = load_model('fruits_model.h5')  # Cambia por tu modelo real
CLASS_NAMES = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"]

# Configurar cámara
camera = cv2.VideoCapture(0)
current_prediction = ""
current_class = ""
current_confidence = 0.0

# Colores para cada fruta (en formato BGR) - ordenados según CLASS_NAMES
fruit_colors = {
    "banana": (0, 255, 255),     # Amarillo
    "fresa": (0, 0, 255),        # Rojo
    "kiwi": (0, 128, 0),         # Verde oscuro
    "manzana": (0, 0, 255),      # Rojo
    "naranja": (0, 165, 255),    # Naranja
    "pina": (0, 255, 255),       # Amarillo
    "sandia": (0, 0, 139),       # Rojo oscuro
    "uva": (128, 0, 128)         # Morado
}

def predict_image(image):
    """Realiza la predicción en una imagen"""
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    
    preds = model.predict(image)[0]
    idx = np.argmax(preds)
    return CLASS_NAMES[idx], float(preds[idx]), preds.tolist()

def add_fruit_effect(frame, fruit_type):
    """Añade efecto de borde y texto según la fruta detectada"""
    color = fruit_colors.get(fruit_type, (255, 255, 255))  # Blanco por defecto
    
    # Añadir borde difuminado
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, 30)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Añadir texto descriptivo
    text = f"Fruta: {fruit_type.upper()}"
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    cv2.putText(frame, text, (text_x, frame.shape[0] - 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    return frame

def generate_frames():
    """Genera los frames de video con las predicciones"""
    global current_prediction, current_class, current_confidence
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Hacer predicción cada 15 frames
            if cv2.getTickCount() % 15 == 0:
                current_class, current_confidence, _ = predict_image(frame.copy())
                current_prediction = f"{current_class} ({current_confidence*100:.1f}%)"
            
            # Aplicar efectos visuales si se detecta una fruta
            if current_class in fruit_colors:
                frame = add_fruit_effect(frame, current_class)
            
            # Mostrar predicción en el frame
            cv2.putText(frame, current_prediction, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Clasificador de Frutas</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            margin-bottom: 20px;
        }
        #videoFeed {
            border-radius: 8px;
            max-width: 100%;
            margin: 10px 0;
            box-shadow: 0 0 20px rgba(0,0,0,0.2);
        }
        .legend {
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 10px;
            margin: 20px 0;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 0 10px;
        }
        .legend-color {
            width: 20px;
            height: 20px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .progress-container {
            margin: 15px 0;
        }
        .progress-bar {
            height: 25px;
            background: #e0e0e0;
            border-radius: 5px;
            margin: 5px 0;
            overflow: hidden;
            position: relative;
        }
        .progress-fill {
            height: 100%;
            transition: width 0.3s;
        }
        .progress-label {
            position: absolute;
            width: 100%;
            text-align: center;
            line-height: 25px;
            color: #333;
            font-weight: bold;
        }
        .instructions {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
            text-align: left;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Clasificador de Frutas</h1>
        
        <div class="instructions">
            <h3>Frutas reconocidas:</h3>
            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFFF00;"></div>
                    <span>Banana</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FF0000;"></div>
                    <span>Fresa</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #008000;"></div>
                    <span>Kiwi</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FF0000;"></div>
                    <span>Manzana</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFA500;"></div>
                    <span>Naranja</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #FFFF00;"></div>
                    <span>Piña</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #8B0000;"></div>
                    <span>Sandía</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background-color: #800080;"></div>
                    <span>Uva</span>
                </div>
            </div>
        </div>
        
        <img id="videoFeed" src="{{ url_for('video_feed') }}">
        
        <div class="result-box">
            <h2>Predicción actual: <span id="predictionText">Acercando fruta...</span></h2>
            <div class="progress-container" id="progressBars"></div>
        </div>
    </div>

    <script>
        // Colores para las barras de progreso (en formato HEX) - ordenados según CLASS_NAMES
        const classColors = {
            "banana": "#FFFF00",
            "fresa": "#FF0000",
            "kiwi": "#008000",
            "manzana": "#FF0000",
            "naranja": "#FFA500",
            "pina": "#FFFF00",
            "sandia": "#8B0000",
            "uva": "#800080"
        };

        function updateProgressBars(probs) {
            const container = document.getElementById('progressBars');
            container.innerHTML = '';
            
            // Ordenar según CLASS_NAMES
            const sortedClasses = ["banana", "fresa", "kiwi", "manzana", "naranja", "pina", "sandia", "uva"];
            
            for (const className of sortedClasses) {
                if (probs[className] !== undefined) {
                    const percentage = Math.round(probs[className] * 100);
                    const color = classColors[className] || '#cccccc';
                    
                    const barDiv = document.createElement('div');
                    barDiv.className = 'progress-bar';
                    
                    const fillDiv = document.createElement('div');
                    fillDiv.className = 'progress-fill';
                    fillDiv.style.width = `${percentage}%`;
                    fillDiv.style.background = color;
                    
                    const labelDiv = document.createElement('div');
                    labelDiv.className = 'progress-label';
                    labelDiv.textContent = `${className}: ${percentage}%`;
                    
                    barDiv.appendChild(fillDiv);
                    barDiv.appendChild(labelDiv);
                    container.appendChild(barDiv);
                }
            }
        }

        // Actualizar predicción cada segundo
        async function updatePrediction() {
            try {
                const response = await fetch('/get_prediction');
                const data = await response.json();
                
                if (data.class) {
                    document.getElementById('predictionText').textContent = 
                        `${data.class} (${Math.round(data.confidence * 100)}%)`;
                    updateProgressBars(data.probabilities);
                }
            } catch (error) {
                console.error('Error:', error);
            }
        }

        setInterval(updatePrediction, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), 
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    _, frame = camera.read()
    if frame is not None:
        class_name, confidence, probabilities = predict_image(frame)
        return {
            "class": class_name,
            "confidence": confidence,
            "probabilities": dict(zip(CLASS_NAMES, probabilities))
        }
    return {"error": "No se pudo capturar frame"}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)