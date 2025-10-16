# modelo.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from skimage import color

class ModeloCNN:
    def __init__(self, num_classes=10, input_shape=(32, 32, 1)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model = None

    def construir(self):
        """Construye la arquitectura de la CNN"""
        self.model = Sequential([
            Input(shape=self.input_shape),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            MaxPooling2D((2,2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.4),
            Dense(self.num_classes, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def entrenar(self, x_train, y_train, epochs=15, batch_size=64):
        """Entrena el modelo con los datos proporcionados"""
        self.model.fit(x_train, y_train, epochs=epochs,
                       batch_size=batch_size, validation_split=0.1)

    def guardar(self, ruta):
        """Guarda el modelo en un archivo .h5"""
        self.model.save(ruta)

    def cargar(self, ruta):
        """Carga un modelo guardado"""
        self.model = load_model(ruta)

    def preprocesar_imagen(self, img):
        """Convierte imagen RGB a gris, la redimensiona y la adapta al input del modelo"""
        from skimage.transform import resize
        if img.shape[-1] == 3:  # si tiene canales RGB
            img = color.rgb2gray(img)
        img = resize(img, (self.input_shape[0], self.input_shape[1]))
        img = img.reshape(1, *self.input_shape)
        return img

    def predecir(self, imagen):
        """Realiza la predicción y devuelve la clase"""
        pred = self.model.predict(imagen)
        return np.argmax(pred, axis=1)[0]
