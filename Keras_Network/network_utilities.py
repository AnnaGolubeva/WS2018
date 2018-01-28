# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 11:08:34 2018

@author: Stavros
"""

### Various network utilities ###

import numpy as np
from os import path, makedirs
from keras.models import Sequential, save_model
from keras.layers import Dense, InputLayer
from keras.callbacks import History, LearningRateScheduler
import keras.backend as K


def create_network(d_in, d_out, hidden_architecture, hidden_activation, out_activation):
    """Returns a keras model with the given architecture"""
    model = Sequential()
    # Input layer
    model.add(InputLayer(input_shape=(d_in,)))
    # Hidden layers
    for i in hidden_architecture:
        model.add(Dense(i, activation=hidden_activation))
        # Output layer
    model.add(Dense(d_out, activation=out_activation))
    
    return model

###################################
####### Callback functions ########
###################################
    
def get_metrics(metrics):
    """Gets the keras dictionary metrics and returns them in an array"""
    metrics_keys = ['loss', 'val_loss', 'acc', 'val_acc']
    k = len(metrics_keys)
    
    metrics_array = []
    for key in range(k):
        metrics_array.append(metrics[0][metrics_keys[key]])
        
    for metric_dict in metrics[1:]:
        for key in range(k):
            metrics_array[key] += metric_dict[metrics_keys[key]]            
    
    return np.array(metrics_array)
    
class callback_class():
    def __init__(self):
        # List with different decay functions
        self.decay_choice_list = [None, self.scheduler_lin]
    
    def create_callbacks(self, model, decay_choice, n_epoch):
        """Returns a callback list for training
        According to the choice, it includes a decay rate schedule"""    
        callbacks = [History()]
        choice = self.decay_choice_list[decay_choice]
    
        if choice == None:
            return callbacks
    
        self.learn_rate = K.eval(model.optimizer.lr)
        self.n_epoch = n_epoch
        self.model = model
        callbacks.append(LearningRateScheduler(choice))
        return callbacks 
    
    def scheduler_lin(self, epoch):
        #if current_rate > 0.01:
        #K.set_value(model.optimizer.lr, learn_rate * np.exp(- epoch * decay_rate))
        K.set_value(self.model.optimizer.lr, self.learn_rate * (1 - epoch / self.n_epoch))
        return K.eval(self.model.optimizer.lr)

############################################
########## Save networks and data ##########
############################################
        
def create_directory(dir_name):
    """Creates directory if it doesn't exist"""
    if not path.exists(dir_name):
        makedirs(dir_name)
        
def create_data_directories():
    create_directory('Metrics/')
    create_directory('MIvalues/')
    

#########################################
########## Get layer functions ##########
#########################################
    
def layer_functions(model):
    """Returns a list of keras functions that read layer outputs"""
    get_layer = []
    for i in range(1, len(model.layers)):
        get_layer.append(kfunc([model.layers[0].input], [model.layers[i].output]))
    
    return get_layer

def get_all_layers(layer_functions, x):
    """Applies the given layer functions in input x to extract output of layers"""
    outputs = []
    for f in layer_functions:
        outputs.append(f([x])[0])
    return outputs
   

