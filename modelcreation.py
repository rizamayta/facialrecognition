import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf                       
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential       #To create the sequential layer

from keras.layers.core import Flatten, Dense, Dropout     #To create the model
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D  #To create the model
from tensorflow.keras.preprocessing import image             #used for image classification
from tensorflow.keras.preprocessing.image import ImageDataGenerator  #used to expand the training dataset in order to improve the performance and ability of the model to generalize

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam          #To use the optimizer
from keras.utils import np_utils  
from tensorflow.keras.utils import plot_model
from IPython.display import SVG, Image

dataofemotion = pd.read_csv('fer2013.csv')
dataofemotion.head()

X_train = []
y_train = []
X_test = []
y_test = []
for index, row in dataofemotion.iterrows():
    k = row['pixels'].split(" ")
    if row['Usage'] == 'Training':
        X_train.append(np.array(k))
        y_train.append(row['emotion'])
    elif row['Usage'] == 'PublicTest':
        X_test.append(np.array(k))
        y_test.append(row['emotion'])

#--------------------Converting Lists to Numpy arrays------------------------------
X_train = np.array(X_train, 'float32')
y_train = np.array(y_train, 'float32')
X_test = np.array(X_test,'float32')
y_test = np.array(y_test,'float32')

#-----------------------Reshaping Pixels arrays---------------------------------
#normalizing data between o and 1  
X_train -= np.mean(X_train, axis=0)  
X_train /= np.std(X_train, axis=0)  

X_test -= np.mean(X_test, axis=0)  
X_test /= np.std(X_test, axis=0) 

#reshape the numpy array to be passed to the model
X_train = X_train.reshape(X_train.shape[0], 48, 48, 1)   
X_test = X_test.reshape(X_test.shape[0], 48, 48, 1)

print(X_test.shape)
print(type(X_test))
print(X_train.shape)

#------------------Convert Labels array to categorial ones---------------------
#y_train= tf.keras.utils.to_categorical(y_train, num_classes=7)
#y_test = tf.keras.utils.to_categorical(y_test, num_classes=7)

y_train= to_categorical(y_train,7)
y_test = to_categorical(y_test,7)
print(y_train)
print(y_train.shape)
print(type(y_train))

model1 = Sequential()

model1.add(ZeroPadding2D((1,1),input_shape=(48,48,1)))
model1.add(Convolution2D(64, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(64, 3, 3, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding="same"))

model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(128, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(128, 3, 3, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding="same"))

model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(256, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(256, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(256, 3, 3, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding="same"))

model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(MaxPooling2D(pool_size=(2,2), strides=(2,2),padding="same"))


model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(ZeroPadding2D((1,1)))
model1.add(Convolution2D(512, 3, 3, activation='relu'))
model1.add(MaxPooling2D((2,2), strides=(2,2),padding="same"))

model1.add(Flatten())
model1.add(Dense(4096, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(4096, activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(7, activation='softmax'))
model1.summary()


model1.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])

batch = 32
epoch = 30

history = model1.fit(X_train,y_train,batch_size= batch,epochs= epoch,verbose=1,validation_data=(X_test, y_test),shuffle=True)


loss_and_metrics = model1.evaluate(X_test,y_test)
print(loss_and_metrics)


plot_model(model1, to_file='model.png', show_shapes=True, show_layer_names=True)
Image('model.png',width=400, height=200)


model_json = model1.to_json()
with open("trainingmodel1.json", "w") as json_file:
  json_file.write(model_json)
  model.save_weights("trainingmodel1.h5")

print("Saved model to disk")