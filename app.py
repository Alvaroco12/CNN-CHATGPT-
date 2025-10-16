# app.py
from flask import Flask, request, render_template
from modelo import ModeloCNN
from skimage import color
from skimage.transform import resize
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

# 🧠 1. Cargar modelo entrenado
modelo = ModeloCNN()
modelo.cargar("modelo_cnn.h5")

# Etiquetas de las clases CIFAR-10
CLASES = ["avión", "auto", "pájaro", "gato", "ciervo", "perro", "rana", "caballo", "barco", "camión"]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'imagen' not in request.files:
        return "No se ha enviado ninguna imagen."

    file = request.files['imagen']

    # 📸 Leer imagen
    img = plt.imread(file)

    # 🧼 Preprocesar igual que en el entrenamiento
    if img.shape[-1] == 4:  # quitar canal alpha si existe
        img = img[:, :, :3]

    img_gray = color.rgb2gray(img)
    img_resized = resize(img_gray, (32, 32))
    img_ready = img_resized.reshape(1, 32, 32, 1)

    # 🧠 Predicción
    pred = modelo.predecir(img_ready)
    clase_predicha = CLASES[pred]

    return f"La clase predicha es: {clase_predicha} ({pred})"

if __name__ == "__main__":
    app.run(debug=True)
