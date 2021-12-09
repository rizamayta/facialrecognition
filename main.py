import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd                       #reading, writing and manipulating the data (using tables)
import numpy as np                        #Library for linear algebra and some probabiltity (raw data)
import os
from keras.preprocessing import image 
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import webbrowser
json_file = open('trainingmodel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load weights into new model
model.load_weights("trainingmodel.h5")
print("Loaded model from disk")
#image loading and converting into grayscale to precit according to the model
img_directory ='gwarasmile.jpg'
img_data = image.load_img(img_directory, target_size = (48, 48))   #load the image from the directory
img_data = image.img_to_array(img_data)                            #convert the image to a Numpy array
img_data = tf.image.rgb_to_grayscale(img_data)

  #print(img_data.shape)
  #img_data = np.array(img_data, 'float32')
  #img_data.resize(48,48,1)
  #print(img_data.shape)
img_data = np.expand_dims(img_data, axis = 0)                     #expands the array by inserting a new axis at the specified position.
  #print(img_data.shape)
classify = model.predict(img_data)
print(classify)
display(Image(img_directory,width= 300, height=300))
print("\n")
max_index = np.argmax(classify[0])
emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
emotion_prediction = emotion_detection[max_index]  
print(max_index)
print(emotion_prediction)
# song playing
import webbrowser
if (emotion_prediction== 'happy'):
    webbrowser.open('https://www.youtube.com/watch?v=A-sfd1J8yX4')# linking playing 
elif (emotion_prediction=='angry'):
    webbrowser.open('')
else:
    webbrowser.open('file://' + os.path.realpath('mix.m4a'))#audio in thefile played