import cv2
import os
from random import shuffle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as pltpip
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import pandas as pd

#dataPath = '/Kaggle'
training_path_aut = 'Kaggle-20210601T155419Z-001/Kaggle/train/train/autistic'
training_path_non_aut = 'Kaggle-20210601T155419Z-001/Kaggle/train/train/non_autistic'

testing_path = 'Kaggle-20210601T155419Z-001/Kaggle/test/test'

learningRate = 0.001
image_size = 100

modelName = 'autisticVSnon_autistic'


def createName(imageName):
    label = imageName.split(' ')[0]
    if label == 'autistic':
        return np.array([1, 0])
    elif label == 'non_autistic':
        return np.array([0, 1])


def create():
    trainData = []
    # Training Data Autistic
    for img in os.listdir(training_path_aut):
        path = os.path.join(training_path_aut, img)
        image = cv2.imread(path, 0)
        image = cv2.resize(image, (image_size, image_size))
        trainData.append([np.array(image), createName(img)])
    # Training Data Non-Autistic
    for img in os.listdir(training_path_non_aut):
        path = os.path.join(training_path_non_aut, img)
        image = cv2.imread(path, 0)
        image = cv2.resize(image, (image_size, image_size))
        trainData.append([np.array(image), createName(img)])

    shuffle(trainData)
    np.save('train.npy', trainData)

    return trainData


def create_test():
    trainData = []
    # Training Data Autistic
    for img in os.listdir(testing_path):
        path = os.path.join(testing_path, img)
        image = cv2.imread(path, 0)
        image = cv2.resize(image, (image_size, image_size))
        trainData.append([np.array(image)])

    np.save( 'test.npy', trainData)

    return trainData


training_Data_path ='train.npy'
testing_Data_path =  'test.npy'

# train
if os.path.exists(training_Data_path):
    trainingData = np.load(training_Data_path)
else:
    trainingData = create()

# test
if os.path.exists(testing_Data_path):
    testingData = np.load(testing_Data_path)
else:
    testingData = create_test()

# First Model
traindata = trainingData
testdata = testingData

shuffle(traindata)
X_train = np.array([i[0] for i in traindata]).reshape(-1, image_size, image_size, 1)
Y_train = np.array([i[1] for i in traindata])

X_Validation = X_train[2122:2221, :]
Y_Validation = Y_train[2122:2221, :]

X_train = X_train[0:2122, :]
Y_train = Y_train[0:2122, :]

X_test = np.array([i[0] for i in testingData]).reshape(-1, image_size, image_size, 1)

#tf.rest_default_graph()
conv_input = input_data(shape=[None, image_size, image_size, 1], name='input')
conv1 = conv_2d(conv_input, 96, 11, activation='relu')
conv2 = conv_2d(conv1, 256, 5, activation='relu')
pool = max_pool_2d(conv2, 5)
conv3 = conv_2d(pool, 384, 3, activation='relu')
pooltwo = max_pool_2d(conv3, 5)
conv4 = conv_2d(pooltwo, 384, 3, activation='relu')
conv5 = conv_2d(conv4, 256, 3, activation='relu')
poolthree = max_pool_2d(conv5, 5)

firstfullyconnected = fully_connected(poolthree, 4096, activation='relu')
secondfullyconnected = fully_connected(firstfullyconnected, 4096, activation='relu')
thirdfullyconnected = fully_connected(secondfullyconnected, 1000, activation='relu')

cnn = fully_connected(thirdfullyconnected, 2, activation='softmax')

cnn1 = regression(cnn, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn1, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=10,
              validation_set=({'input': X_Validation}, {'targets': Y_Validation}), snapshot_step=500,
              show_metric=True, run_id=modelName)
    model.save('model.tfl')

for img in os.listdir(testing_path):
    path = os.path.join(testing_path, img)
    image = cv2.imread(path, 0)
    image = cv2.resize(image, (image_size, image_size))
    pr = model.predict([image])[0]

    print(pr[0])
    print(pr[1])

############################################################################################
conv_input = input_data(shape=[None, image_size, image_size, 1], name='input')
conv1 = conv_2d(conv_input, 64, 3, activation='relu')
conv2 = conv_2d(conv1, 64, 3, activation='relu')
pool = max_pool_2d(conv2, 2,strides=2)


conv3 = conv_2d(pool, 128, 3, activation='relu')
conv4 = conv_2d(conv3, 128, 3, activation='relu')
pooltwo = max_pool_2d(conv4, 2,strides=2)



conv5 = conv_2d(pooltwo, 256, 3, activation='relu')
conv6 = conv_2d(conv5, 256, 3, activation='relu')
poolthree = max_pool_2d(conv6, 2,strides=2)

conv7 = conv_2d(poolthree, 512, 3, activation='relu')
conv8 = conv_2d(conv7, 512 , 3, activation='relu')
conv9 = conv_2d(conv8, 512 , 3, activation='relu')
poolfour= max_pool_2d(conv9, 2,strides=2)

conv10 = conv_2d(poolfour, 512, 3, activation='relu')
conv11 = conv_2d(conv10, 512 , 3, activation='relu')
conv12 = conv_2d(conv11, 512 , 3, activation='relu')
poolfive= max_pool_2d(conv12, 2,strides=2)


thirdfullyconnected = fully_connected(poolfive, 1000, activation='relu')
thirdfullyconnected=dropout(thirdfullyconnected,.5)

cnn = fully_connected(thirdfullyconnected, 2, activation='softmax')

cnn1 = regression(cnn, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn1, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=2,
              validation_set=({'input': X_Validation}, {'targets': Y_Validation}), snapshot_step=500,
              show_metric=True, run_id=modelName)
    model.save('model.tfl')


names=[]
out=[]
for img in os.listdir(testing_path):
    path = os.path.join(testing_path, img)
    image = cv2.imread(path, 0)
    image = cv2.resize(image, (image_size, image_size))
    names.append(img)

    pr = model.predict([image])[0]
    if(pr[0]>pr[1]):
        out.append(1)
    else :
        out.append(0)


dict = {'Image': names, 'Label': out}
df = pd.DataFrame(dict)
df.to_csv('Submit.csv')








###################################################################VGG19#########

conv_input = input_data(shape=[None, image_size, image_size, 1], name='input')
conv1 = conv_2d(conv_input, 64, 3, activation='relu')
conv2 = conv_2d(conv1, 64, 3, activation='relu')
pool = max_pool_2d(conv2, 2,strides=2)


conv3 = conv_2d(pool, 128, 3, activation='relu')
conv4 = conv_2d(conv3, 128, 3, activation='relu')
pooltwo = max_pool_2d(conv4, 2,strides=2)



conv5 = conv_2d(pooltwo, 256, 3, activation='relu')
conv6 = conv_2d(conv5, 256, 3, activation='relu')
poolthree = max_pool_2d(conv6, 2,strides=2)

conv7 = conv_2d(poolthree, 512, 3, activation='relu')
conv8 = conv_2d(conv7, 512 , 3, activation='relu')
conv9 = conv_2d(conv8, 512 , 3, activation='relu')
conv10 = conv_2d(conv9, 512 , 3, activation='relu')

poolfour= max_pool_2d(conv10, 2,strides=2)

conv11 = conv_2d(poolfour, 512, 3, activation='relu')
conv12 = conv_2d(conv11, 512 , 3, activation='relu')
conv13 = conv_2d(conv12, 512 , 3, activation='relu')
conv14 = conv_2d(conv13, 512 , 3, activation='relu')

poolfive= max_pool_2d(conv14, 2,strides=2)

firstfullyconnected = fully_connected(poolfive, 4096, activation='relu')
firstfullyconnected=dropout(firstfullyconnected,.5)
secondfullyconnected = fully_connected(firstfullyconnected, 4096, activation='relu')
secondfullyconnected=dropout(secondfullyconnected,0.5)

thirdfullyconnected = fully_connected(secondfullyconnected, 1000, activation='relu')

thirdfullyconnected=dropout(thirdfullyconnected,.7)

cnn = fully_connected(thirdfullyconnected, 2, activation='softmax')

cnn1 = regression(cnn, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(cnn1, tensorboard_dir='log', tensorboard_verbose=3)

if os.path.exists('model.tfl.meta'):
    model.load('./model.tfl')
else:
    model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=2,
              validation_set=({'input': X_Validation}, {'targets': Y_Validation}), snapshot_step=500,
              show_metric=True, run_id=modelName)
    model.save('model.tfl')


names=[]
out=[]
for img in os.listdir(testing_path):
    path = os.path.join(testing_path, img)
    image = cv2.imread(path, 0)
    image = cv2.resize(image, (image_size, image_size))
    names.append(img)

    pr = model.predict([image])[0]
    if(pr[0]>pr[1]):
        out.append(1)
    else :
        out.append(0)


dict = {'Image': names, 'Label': out}
df = pd.DataFrame(dict)
df.to_csv('Submit.csv')


def identity_block(X,f,filter):
    f1,f2,f3=filter
    X_copy=X
    X=conv_2d(X,f1,1,activation='relu')
    X=conv_2d(X,f1,f,activation='relu')
    X=conv_2d(X,f1,1,activation='relu')
    X = np.concatenate(X, X_copy)
    return X

def convolutional_block(X, f, filters):
    f1, f2, f3 = filters
    X_copy = X
    X = conv_2d(X, f1, 1, activation='relu',strides=2)
    X = conv_2d(X, f1, f, activation='relu')
    X = conv_2d(X, f1, 1, activation='relu')
    X_copy=conv_2d(X_copy,f3,1,activation='relu',strides=2)
    X=np.concatenate(X,X_copy)
    return X


def ResNet50(image):


    image=conv_2d(image, 64, 7, activation='relu',strides=2)
    image = max_pool_2d(image,3, strides=2)

    X = convolutional_block(image, f=3, filters=[64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    # Stage 3
    X = convolutional_block(X, f=3, filters=[128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    # Stage 4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    # Stage 5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    # AVGPOOL.
    X = max_pool_2d(X,2)

    # output layer
    cnn = fully_connected(X, 2, activation='softmax')

    cnn1 = regression(cnn, optimizer='adam', learning_rate=learningRate, loss='mean_square',
                      name='targets')

    model = tflearn.DNN(cnn1, tensorboard_dir='log', tensorboard_verbose=3)

    if os.path.exists('model.tfl.meta'):
        model.load('./model.tfl')
    else:
        model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=2,
                  validation_set=({'input': X_Validation}, {'targets': Y_Validation}), snapshot_step=500,
                  show_metric=True, run_id=modelName)
        model.save('model.tfl')

















