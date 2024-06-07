from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import random
from subprocess import check_output
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import cv2
import os
import warnings
# filter warnings
warnings.filterwarnings('ignore')
print(os.listdir("./input"))
print(check_output(["ls", "./input"]).decode("utf8"))


# LOADING DATA

np.random.seed(1234)
directory = "/Users/mauriciovargas/Downloads/archive/fruits-360_dataset/fruits-360/Training"
classes = ["Apple Golden 1", "Avocado", "Banana", "Cherry 1", "Cocos", "Kiwi",
           "Lemon", "Mango", "Orange"]

all_arrays = []
img_size = 100
for i in classes:
    path = os.path.join(directory, i)
    class_num = classes.index(i)
    for img in os.listdir(path):
        # img_array=cv2.imread(os.path.join(path,img),
        #                     cv2.IMREAD_GRAYSCALE)
        img_array = cv2.imread(os.path.join(path, img))
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        # img_array=cv2.resize(img_array,(img_size,img_size))
        all_arrays.append([img_array, class_num])


# ----------------------------------
directory2 = "/Users/mauriciovargas/Downloads/archive/fruits-360_dataset/fruits-360/Training"
classes2 = ["Apple Golden 1", "Avocado", "Banana", "Cherry 1", "Cocos", "Kiwi",
            "Lemon", "Mango", "Orange"]

all_arrays2 = []
img_size = 100
for i in classes2:
    path = os.path.join(directory2, i)
    class_num2 = classes2.index(i)
    for img in os.listdir(path):
        # img_array2=cv2.imread(os.path.join(path,img),
        #                     cv2.IMREAD_GRAYSCALE)
        img_array2 = cv2.imread(os.path.join(path, img))
        img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_BGR2RGB)
        # img_array2=cv2.resize(img_array2,(img_size,img_size))
        all_arrays2.append([img_array2, class_num2])
# ----------------------------------

fruits_array_train = []
for features, label in all_arrays:
    fruits_array_train.append(features)

location = [[1, 500, 1150], [1500, 2000, 2500], [3000, 3500, 4000]]
fruit_names = ["Apple", "Avocado", "Banana", "Cherry",
               "Cocos", "Kiwi", "Lemon", "Mango", "Orange"]
a = 0
b = 1
c = 2
for i, j, k in location:
    plt.subplots(figsize=(8, 8))
    plt.subplot(1, 3, 1)
    plt.imshow(fruits_array_train[i])
    plt.title(fruit_names[a])
    plt.axis("off")
    plt.subplot(1, 3, 2)
    plt.imshow(fruits_array_train[j])
    plt.title(fruit_names[b])
    plt.axis("off")
    plt.subplot(1, 3, 3)
    plt.imshow(fruits_array_train[k])
    plt.title(fruit_names[c])
    plt.axis("off")
    a += 3
    b += 3
    c += 3

# 🧬 Creating Features 🧬

random.shuffle(all_arrays)

X_train = []
Y_train = []
for features, label in all_arrays:
    X_train.append(features)
    Y_train.append(label)
X_train = np.array(X_train)  # arraying

random.shuffle(all_arrays2)

X_test = []
Y_test = []
for features, label in all_arrays2:
    X_test.append(features)
    Y_test.append(label)
X_test = np.array(X_test)  # arraying

# 🧾 Normalization 🧾
# normalization and reshaping
X_train = X_train.reshape(-1, img_size, img_size, 3)
X_train = X_train/255
X_test = X_test.reshape(-1, img_size, img_size, 3)
X_test = X_test/255
print("shape of X_train= ", X_train.shape)
print("shape of X_test=  ", X_test.shape)

# 🧾 Comverting Into Categorical Form 🧾
Y_train = to_categorical(Y_train, num_classes=9)
Y_test = to_categorical(Y_test, num_classes=9)

#################
Y_train.shape
X_train.shape

# 🧾 Train-Test-Split 🧾
x_train, x_val, y_train, y_val = train_test_split(
    X_train, Y_train, test_size=0.2, random_state=42)

# 🛠 Model Architecture 🛠


model = Sequential()
model.add(Conv2D(filters=8, kernel_size=(3, 3), padding="Same",
          activation="relu", input_shape=(100, 100, 3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(filters=16, kernel_size=(3, 3),
          padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.35))

model.add(Conv2D(filters=32, kernel_size=(3, 3),
          padding="Same", activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.35))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(9, activation="softmax"))

# Defining optimizer
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

# Compile the model
model.compile(optimizer=optimizer,
              loss="categorical_crossentropy", metrics=["accuracy"])

epochs = 10
batch_size = 18

# 🧬 Model Training 🧬
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=False,
    vertical_flip=False
)

# Assuming x_train and y_train are defined
datagen.fit(x_train)

# Model fitting using `fit` method
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(x_val, y_val),
    steps_per_epoch=x_train.shape[0] // batch_size
)
# 💹 Model Training Graph 📈
plt.plot(history.history["val_accuracy"], color="r", label="val_acc")
plt.title("Accuracy Graph")
plt.xlabel("number of epochs")
plt.ylabel("accuracy")
plt.legend()
plt.grid()
plt.show()

# 🧾 Confusion Matrix 🧾
y_pred = model.predict(x_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)
# compute conf mat
conf_mat = confusion_matrix(y_true, y_pred_classes)
# plot the con mat
fruit_names = ["Apple", "Avocado", "Banana", "Cherry",
               "Cocos", "Kiwi", "Lemon", "Mango", "Orange"]
f, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(conf_mat, annot=True, fmt='.0f')
ax.set_xticklabels(fruit_names)
ax.set_yticklabels(fruit_names)
plt.show()


# We can see error values on validation data

# confusion matrix
y_pred2 = model.predict(X_test)
y_pred_classes2 = np.argmax(y_pred2, axis=1)
y_true2 = np.argmax(Y_test, axis=1)
# compute conf mat
conf_mat2 = confusion_matrix(y_true2, y_pred_classes2)
# plot the con mat
f, ax = plt.subplots(figsize=(10, 9))
sns.heatmap(conf_mat2, annot=True, fmt=".0f")
ax.set_xticklabels(fruit_names)
ax.set_yticklabels(fruit_names)
plt.show()

# exportar modelo
model.save('fruit_classification_model.h5')
