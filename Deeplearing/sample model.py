import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Add, Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, \
    MaxPool2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import Dense, Dropout, Flatten, Activation, Concatenate, Lambda
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, Adamax
from tensorflow.keras import regularizers, activations
from keras.preprocessing import image
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

Image_size = 100
batch = 32
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

# Proving the path of training and test dataset
# Setting the image input size as (224, 224)
# We are using class mode as binary because there are only two classes in our data
training_set = train_datagen.flow_from_directory("/content/gdrive/MyDrive/Kaggle/train/train",
                                                 target_size=(Image_size, Image_size),
                                                 batch_size=batch,
                                                 shuffle=True,
                                                 )





CNN_Model = Sequential()


CNN_Model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(image_size, image_size, 1)))

CNN_Model.add(Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
CNN_Model.add(MaxPool2D(pool_size=(2,2)))

CNN_Model.add(Conv2D(256, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
CNN_Model.add(MaxPool2D(pool_size=(2,2)))

CNN_Model.add(Conv2D(512, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu'))
CNN_Model.add(MaxPool2D(pool_size=(2,2)))


CNN_Model.add(Flatten())


# hidden layer
CNN_Model.add(Dense(units=500, activation='relu'))
CNN_Model.add(Dropout(0.4))
CNN_Model.add(Dense(units=250, activation='relu'))
CNN_Model.add(Dropout(0.3))

# output layer
CNN_Model.add(Dense(units=2, activation='softmax'))
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=20)
mc = ModelCheckpoint('/content/gdrive/My Drive/best_model.h5', monitor='val_accuracy', mode='max')

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy',
                             verbose=1, save_best_only=True, mode='max')

early = EarlyStopping(monitor="val_loss",
                      mode="min",
                      patience=4, restore_best_weights=True)

sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=False)


CNN_Model.compile(optimizer=Adamax(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

CNN_Model.fit(x=training_set, epochs=100, callbacks=[checkpoint, early])

images = []
images_name = []
r = []
test_path = '/content/gdrive/MyDrive/Kaggle/test/test'

for img in os.listdir(test_path):
    images_name.append(img)
    img = os.path.join("/content/gdrive/MyDrive/Kaggle/test/test", img)
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

dataset.to_csv(r'predictions.csv', index=False)

dataset.Label.value_counts()