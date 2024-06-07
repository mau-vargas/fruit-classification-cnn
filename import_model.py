from tensorflow.keras.models import load_model
import cv2
import numpy as np


def prepare_image(image_path, img_size=100):
    # Cargar la imagen
    img = cv2.imread(image_path)
    # Redimensionar la imagen
    img = cv2.resize(img, (img_size, img_size))
    # Convertir la imagen a RGB (si no está ya en ese formato)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalizar los valores de los píxeles (si entrenaste el modelo con imágenes normalizadas)
    img = img / 255.0
    # Añadir una dimensión extra para que el modelo lo acepte (batch size, height, width, channels)
    img = np.expand_dims(img, axis=0)
    return img


# Cargar el modelo
model = load_model('fruit_classification_model.h5')

# Ruta de la imagen de prueba
# image_path = '/Users/mauriciovargas/Downloads/archive/fruits-360_dataset/fruits-360/Test/Apricot/3_100.jpg'
# image_path = '/Users/mauriciovargas/Downloads/naranja_1.jpeg'
image_path = '/Users/mauriciovargas/Desktop/palta_2.png'

# Preparar la imagen
prepared_image = prepare_image(image_path)

# Hacer una predicción
prediction = model.predict(prepared_image)

# Obtener la clase con la mayor probabilidad
predicted_class = np.argmax(prediction, axis=1)

# Definir las clases
classes = ["Apple Golden 1", "Avocado", "Banana",
           "Cherry 1", "Cocos", "Kiwi", "Lemon", "Mango", "Orange"]

# Obtener la clase predicha
predicted_class_name = classes[predicted_class[0]]

print(f'La imagen es clasificada como: {predicted_class_name}')
