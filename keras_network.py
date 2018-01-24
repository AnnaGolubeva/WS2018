# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:11:26 2018

@author: Stavros
"""

import numpy as np

### Load Dataset ###
confs = np.load('toy_dataset\configs.npy')
labels = np.load('toy_dataset\labels.npy')
# Get dataset sizes
(total_samples, d_input) = confs.shape
n_labels = 2 # for one hot
# Convert to one-hot
labels_hot = np.empty([total_samples, 2])
labels_hot[:,0] = (labels == 0).astype(np.int)
labels_hot[:,1] = (labels == 1).astype(np.int)

### Split dataset and testing dataset ###
train_percentage = 0.85 # percentage of data to use for training (rest are for testing)
# Number of samples in training and testing
train_samples = np.int(train_percentage * total_samples)
test_samples  = total_samples - train_samples
# Randomly choose data for splitting
indices_list = np.arange(total_samples)
np.random.shuffle(indices_list)
# Data for training
train_confs = confs[indices_list[:train_samples]]
train_labels = labels_hot[indices_list[:train_samples]]
# Data for testing
test_confs = confs[indices_list[train_samples:]]
test_labels = labels[indices_list[train_samples:]]

### Training Parameters ###
# Given by user after you run
n_epoch = int(input('Give number of epochs: '))
batch_size = int(input('Give batch size: '))
learn_rate = float(input('Give learning rate: '))
# (Optional: Use decay of learning rate during training)
decay_learning_rate = True

### Network architecture ###
hidden_architecture = [12, 10, 7, 5, 4, 3]  # List with number of neurons of each layer
hidden_activation = 'tanh'  # Activation function of hidden layers
n_hidden = len(hidden_architecture) # Number of hidden layers

### Define model ###
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from keras.optimizers import SGD
from keras.callbacks import History, LearningRateScheduler

model = Sequential()
# Input layer
model.add(InputLayer(input_shape=(d_input,)))
# Hidden layers
for i in hidden_architecture:
    model.add(Dense(i, activation=hidden_activation))
# Output layer
model.add(Dense(n_labels, activation='sigmoid'))

### (Optional: Function for decaying learning rate - can try any type of decay) ###
import keras.backend as K
def scheduler(epoch):
    if epoch % 400 == 0:
        K.set_value(model.optimizer.lr, learn_rate / 2.0)
    return K.eval(model.optimizer.lr)

### Training ###
# Train using SGD and cross-entropy as loss
model.compile(optimizer=SGD(lr = learn_rate), loss='categorical_crossentropy')
callbacks=[History()] # Keras function to keep track of loss during training
if decay_learning_rate: # (Optional: if we select to use decay)
    callbacks.append(LearningRateScheduler(scheduler))

# Start training
hist = model.fit(train_confs, train_labels, batch_size=batch_size, epochs=n_epoch, 
                 verbose=0, callbacks=callbacks)
loss = np.array(hist.history['loss']) # Save an array with loss for every epoch

### Testing ###
# Use trained network to predict labels for test data
# (the argmax is to convert the continuous sigmoid output to 1D label of 0 or 1)
pred_labels = np.argmax(model.predict(test_confs), axis=1)
# Calculate accuracy from the number of correctly classified data
accuracy = np.sum((pred_labels == test_labels).astype(np.int)) / test_samples