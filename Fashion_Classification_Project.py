# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 21:52:46 2019

@author: Hammad
"""

'''
Problem Statement:
    We want to build a model that look at images and can tell us
    exactly what category of dress (shirt, bag, pant) it is from.
    
    An Artificial Intelligence Model based on Deep learning model
    that can classify images into different categories or different classes.
    
'''



# Import necessary libraries #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn
import random

# Load Data #

fashion_train_data = pd.read_csv('Dataset/fashion-mnist_train.csv', sep = ',')
fashion_test_data = pd.read_csv('Dataset/fashion-mnist_test.csv', sep = ',') 

# Explore Data #
fashion_train_data.head()
fashion_test_data.head()

fashion_train_data.shape
fashion_test_data.shape

# Data Pre-processing / Preparation

training = np.array(fashion_train_data, dtype = 'float32')
testing = np.array(fashion_test_data, dtype = 'float32')

# We know our dataset contains images in binary form. Let's visulization one image from training dataset to get the feel 
# what's inside the dataset. E.g random i will pick 600th with all values in the column
i = random.randint(1, 60000)
plt.imshow(training[i, 1:].reshape(28, 28))
label=training[i, 0]
plt.title(label)

# When we are using imshow, we need to reshape our data and since our images
# are 28x28 that's why we will use 

# Let's view more images in grid format
# Define the dimensions of the plot grid

W_grid = 15
L_grid = 15

# fig, axes  = plt.subplots(L_grid, W_grid)
# subplot return the figure object and axes object
# we can use the axes object to plot specific figures at various locations

fig, axes = plt.subplots(L_grid, W_grid, figsize = (17, 17))
axes = axes.ravel()

n_training = len(training) # get the length of training dataset

# Select a random number from 0 to n_training

for i in np.arange(0, W_grid * L_grid):
    # Select random number
    index = np.random.randint(0, n_training)
    
    # Read and display an image with selected index
    axes[i].imshow(training[index, 1:].reshape((28,28)) )
    axes[i].set_title(training[index, 0], fontsize = 8)
    axes[i].axis('off')
    
plt.subplots_adjust(hspace = 0.4)
    
    

# Model Training #

# We will use Convolution Neural Network

# Train our Model

X_train = training[:, 1:]/255
y_train = training[:, 0]

X_test = testing[:, 1:]/255
y_test = testing[:, 0]

# Data Validation Test

from sklearn.model_selection import train_test_split

X_train, X_validate, y_train, y_validate = train_test_split(
                                                            X_train,
                                                            y_train,
                                                            test_size = 0.2,
                                                            random_state = 12345
                                                            )


# We have all the data in array format. We have to reshape it first in 28x28 matrix

X_train = X_train.reshape(X_train.shape[0], *(28, 28, 1))
X_test = X_test.reshape(X_test.shape[0], *(28, 28, 1))
X_validate = X_validate.reshape(X_validate.shape[0], *(28, 28, 1))

X_train.shape
X_test.shape
X_validate.shape

# Building Our Model with Keras #
# Importing Keras #
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


# Building Model  #
cnn_model = Sequential()
cnn_model.add(Conv2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))

# Now max pooling layer in our model

cnn_model.add(MaxPooling2D(pool_size = (2,2)))

# Flatten the Model
cnn_model.add(Flatten())

cnn_model.add(Dense(output_dim = 32, activation = 'relu'))

cnn_model.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Adam Optimiser

cnn_model.compile(loss = 'sparse_categorical_crossentropy', optimizer = Adam(lr=0.001), metrics = ['accuracy'])

# Fit our CNN Model
epochs = 50
cnn_model.fit(X_train,
              y_train,
              batch_size = 512,
              epochs = epochs,
              verbose = 1,
              validation_data = (X_validate, y_validate))

# Evaluate Model #

evaluation = cnn_model.evaluate(X_test, y_test)
print("Accuracy Score: ", evaluation)

predicted_classes = cnn_model.predict_classes(X_test)

print(predicted_classes)

L = 5
W = 5
fig, axes = plt.subplots(L, W, figsize = (12, 12))
axes = axes.ravel()

for i in np.arange(0, L * W):
    axes[i].imshow(X_test[i].reshape(28, 28))
    axes[i].set_title("Prediction Class = {:0.1f}\n True Class = {:0.1f}".format(predicted_classes[i], y_test[i]))
    axes[i].axis('off')
    
plt.subplots_adjust(wspace = 0.5)

# Confusion Matrix to Show Evaluation

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predicted_classes)
plt.figure(figsize = (14, 10))

# Creating Heatmap to show no. of samples has been correctly specified.
sbn.heatmap(cm, annot = True)

# Classification Report
from sklearn.metrics import classification_report

num_classes = 10
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(y_test, predicted_classes, target_names = target_names))

# Final Remarks
# Model achieves good accuracy of 92% but it can be improved. Class # 6 (which is shirt,
# is the worst performer.)

# Tips to improve Model:
## Increase kernals from 32 to 64
## We can use Dropout technique to improve the model as well












