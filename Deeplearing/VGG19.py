import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, MaxPool2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import regularizers, activations
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

Image_size = 150
train_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/Kaggle/train/train',
                                                 target_size=(Image_size, Image_size),
                                                 batch_size=32,
                                                 class_mode='categorical')

CNN_Model = Sequential()
CNN_Model.add(Conv2D(filters=64, kernel_size=3, activation='relu', input_shape=[Image_size, Image_size, 3]))
CNN_Model.add(Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

CNN_Model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

CNN_Model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))

CNN_Model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))
CNN_Model.add(Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'))

CNN_Model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

CNN_Model.add(Flatten())
CNN_Model.add(Dense(units=4096, activation='relu'))
CNN_Model.add(Dense(units=4096, activation='relu'))
CNN_Model.add(Dense(units=1000, activation='relu'))

CNN_Model.add(Dense(units=2, activation='softmax'))

filepath = "model.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
                             verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=4, restore_best_weights=True)
callbacks_list = [checkpoint, early]

sgd = SGD(learning_rate=0.1, decay=1e-6, momentum=0.9, nesterov=True)

CNN_Model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
CNN_Model.fit(x=training_set, epochs=240, callbacks=callbacks_list)
images = []
images_name = []
r = []
test_path = '/content/drive/MyDrive/Kaggle/test/test'

for img in os.listdir(test_path):
    images_name.append(img)
    img = os.path.join("/content/drive/MyDrive/Kaggle/test/test", img)
    img = image.load_img(img, target_size=(Image_size, Image_size))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    images = np.vstack([img])
    classes = CNN_Model.predict(images)
    if classes[0][0] > classes[0][1]:
        r.append(1)
    else:
        r.append(0)

images_name = np.array(images_name)
dataset = pd.DataFrame({'Image': images_name, 'Label': list(r)}, columns=['Image', 'Label'])
print(classes)

dataset.to_csv(r'predictions2.csv', index=False)

dataset.Label.value_counts()