"""Test ImageNet pretrained DenseNet"""

import cv2
import numpy as np
from keras.optimizers import SGD
import keras.backend as K

import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import MobileNet
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam


base_model=model #imports the mobilenet model and discards the last 1000 neuron layer.

# x=base_model.output
# x=GlobalAveragePooling2D()(x)
# x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dense(1024,activation='relu')(x) #dense layer 2
# x=Dense(512,activation='relu')(x) #dense layer 3
# preds=Dense(5,activation='softmax')(x) #final layer with softmax activation


# In[3]:






# We only test DenseNet-121 in this script for demo purpose 

currdir = os.getcwd()
folders = os.listdir(currdir+"/data/test")
correctimages=0
length = 0
# Test pretrained model
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])


for f in folders:
  if f!= 9:
    currdir1 = f
    #print(currdir1)
    images = os.listdir('data/test/'+str(currdir1))
    length=length+len(images)
    #print(length)
    #print (images)
    #break
    for image in images:
      im = cv2.resize(cv2.imread('data/test/'+str(currdir1)+"/"+image), (224, 224)).astype(np.float32)
      #im = cv2.resize(cv2.imread('shark.jpg'), (224, 224)).astype(np.float32)

      # Subtract mean pixel and multiple by scaling constant 
      # Reference: https://github.com/shicai/DenseNet-Caffe
      im[:,:,0] = (im[:,:,0] - 103.94) * 0.017
      im[:,:,1] = (im[:,:,1] - 116.78) * 0.017
      im[:,:,2] = (im[:,:,2] - 123.68) * 0.017

      if K.image_dim_ordering() == 'th':
        # Transpose image dimensions (Theano uses the channels as the 1st dimension)
        im = im.transpose((2,0,1))

        # Use pre-trained weights for Theano backend
        weights_path = 'model/weights.h5'
      else:
      # Use pre-trained weights for Tensorflow backend
        weights_path = 'model/weights.h5'

        # Insert a new dimension for the batch_size
        im = np.expand_dims(im, axis=0)

        

        out = model.predict(im)

        # Load ImageNet classes file
        classes = []
        with open('plantdisease.txt', 'r') as list_:
          for line in list_:
            classes.append(line.rstrip('\n'))
        #print (out)
        #print(classes[np.argmax(out)] + "  " + f)
        #print ('Prediction: '+str(classes[np.argmax(out)]))
        if classes[np.argmax(out)] == f:
          
          correctimages=correctimages+1

print ((correctimages/length)*100)
