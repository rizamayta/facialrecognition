# Facial Recognition System for song suggestion according to mood
The process of detecting face, analyzing it, converting it into data and finding the similarity in the face structure is the main work of facial recognition system
 <br />
In this project, I intended to develop a model who will learn how to recognize the facial structures present in a dataset and use the model for further implementation. Main objective of the project is to train the model to learn the facial structure to find which expression is presnt in the image and use the model to recognition of other images. .

![facial](/facial.jpeg)


## Overview
* In this project, facial recognition system will be trained to recognize facial features.
Will be using VGG16 (Very Deep Convolutional Networks for Large-Scale Image Recognition) which is Convolutional Network for Classification and Detection for training the dataset.
* Agent will learn using CNN, in which the system remembers which action he performed in each state so that the next time he experiences the same state, he will recognize the features of the face.
* Main objective of the project is to train the agent to learn the strategy to recognize facial features (surprise, happy, neutral, angry, disgust and fear images) and suggest songs accordingly.
* Out of pictures given played, CNN model  should be able to detect facial expression in most of the cases.
<br>

#### Tools and libraries used
* Python 3.7
* Ipython
* Pandas
* Matplotlib
* Web Browser
* Keras
* Tensorflow


#### Requirements
Codes are writtern in python and requires python 3.6 + to run.

## Implementation
* Creating the facial recognition model using Convolutional Neural Network (CNN).
![Facial-Expression](/CNN.jpg)
* The concept is that the convolutional layers extract basic, low-level properties that apply across images — such as edges, patterns, and gradients 
* Use strategies: (Model Creation ,Optimization using Adam(Method for Stochastic Optimization, Saving Model and implementing in the other photos for detection). 
* It will involve 4 steps : Model Creation, Adam Method, Saving Model, Using Model for detection and according to the result playing sound.
* We will do 30 iterations to get better accuracy for the recognition.
* After our model has finished the iteration, we save the model in the format of json and hd5
* Now, we use the model saved for recognization of the image send to the system
* According to the expression present we use the youtube playlist for songs suggestion


![Facial-Expression](/screenshot1.png)


## Running code
Instructions on how to run the project:<br>
**Step 1:** Download the zip file or clone the repository <br>
**Step 2:** cd to the directory where your downloaded folder is located.<br>
**Step 3:** run: `pip install -r requirements.txt` in your shell <br>
**Step 4:** open the project folder in spyder <br>
**Step 5:** Download the dataset ('https://www.kaggle.com/deadskull7/fer2013) if you want to create your own model 
**Step 6:** If you want to create the model you can run modelcreation.py but it will take time for model creation. Also if you want to increase the iteration present to make the model perform more iteration you can open the modelcreation.py file and change the epoch value to the respectiver number you want.
**Step 7:** As the model has already been created you can direclty open main.py and run it to get the output

## Results
Facial Recognition <br>
==================== <br>
![Facial-Expression](/Result1.png)
==================== <br><br>
Playing Sound <br>
==================== <br>
![Facial-Expression](/Result2.png)
==================== <br>

