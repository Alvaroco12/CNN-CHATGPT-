# entrenar_modelo.py
from modelo import ModeloCNN
from skimage import color
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf
import matplotlib.pyplot as plt

# 1. Cargar datos CIFAR-10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# 2. Convertir imágenes a escala de grises y darles forma
x_train_gray = color.rgb2gray(x_train).reshape(-1, 32, 32, 1)
x_test_gray = color.rgb2gray(x_test).reshape(-1, 32, 32, 1)

# 3. One-hot encoding de etiquetas
num_classes = 10
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# 4. Crear y construir el modelo CNN
modelo = ModeloCNN(num_classes=num_classes)
modelo.construir()

# 🪄 5. Early Stopping, sirve para parar el entrenamiento si la precision ya no sube mas, para no sobreajustar el modelo
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

# 6. Entrenar el modelo con Early Stopping
history = modelo.model.fit(
    x_train_gray,
    y_train_cat,
    epochs=50,               # máximo de epochs
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

# 7. Guardar modelo entrenado
modelo.guardar("modelo_cnn.h5")
print("✅ Entrenamiento terminado y modelo guardado en modelo_cnn.h5")
print(f"⏸️ Se detuvo tras {early_stop.stopped_epoch} epochs.")

# 📊 8. Graficar la evolución del entrenamiento
plt.figure(figsize=(10,5))

# 📊 9. Evaluar el modelo en el conjunto de prueba
test_loss, test_acc = modelo.model.evaluate(x_test_gray, y_test_cat, verbose=0)
print(f"📌 Precisión final en test: {test_acc:.4f} - Pérdida: {test_loss:.4f}")


# Precisión
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()

# Pérdida
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida')
plt.xlabel('Épocas')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
