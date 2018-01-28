# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 18:41:26 2018

@author: Stavros
"""

######################################################################
## Script that applies the network and calculates MI simultaneously ##
######################################################################

### NOTE: Learning rate decay cannot be used at the moment ###

import numpy as np
import network_utilities as nf
import dataset_functions as df
from mi_functions import MI_class
from keras.optimizers import SGD
from keras.models import load_model
import matplotlib.pyplot as plt    

class MI_New_Model():
    def __init__(self, data_folder, train_percentage, learning_rate, batch_size 
                 hidden_architecture, decay_choice=0):
        """Defines the given (hyper-)parameters inside class"""
        self.data_folder = data_folder
        self.train_percentage = train_percentage
        self.learning_rate = learning_rate
        self.decay_choice = decay_choice
        self.hidden_architecture = hidden_architecture
        self.batch_size = batch_size
        self.n_epoch = 0
    
    def assign_name(self):
        """ Assign a network name that shows the hyperparameters """
        self.network_name = self.data_folder
        self.network_name += '_ep%dlr%.4fdec%dbatch%d'%(self.n_epoch, self.learning_rate, 
                                                      self.decay_choice, self.batch_size)
        
    def load_data(self):
        """Loads the datasets from files"""
        self.train_confs, self.train_labels, self.test_confs, self.test_labels = df.split_dataset(
        train_percentage=self.train_percentage, data_folder=self.data_folder)
        
        self.total_confs = df.load_dataset(data_folder=self.data_folder)[0]
        
    def create_and_compile(self, hid_act='tanh', out_act='sigmoid', 
                           loss='categorical_crossentropy', metrics_sel=['accuracy']):
        """Creates and compiles the Keras model"""
        self.model = nf.create_network(self.train_confs.shape[1], self.train_labels.shape[1], 
                       self.hidden_architecture, hid_act, out_act)
        
        self.model.compile(optimizer=SGD(lr = self.learning_rate), 
                           loss=loss, metrics=metrics_sel)
                
    def initialize_functions(self):
        """Initializes the functions that get output from the layers"""
        self.metrics = []
        self.get_layer = nf.layer_functions(self.model)
        self.n_layers = len(self.model.layers) - 1
        
        # For MI calculation
        self.MI = [] # MI: [n_mi, n_layers, 2 (for X and Y)]
        self.MI_temp = np.empty([self.n_layers, 2])
        self.mi_obj = MI_class(self.data_folder)
        
    def default_initializer(self):
        """Initializes the whole network (loads data, creates and compiles and functions)
        by using the default settings of each function"""
        self.load_data()
        self.create_and_compile()
        self.initialize_functions()
        
    def _train_with_mi_calculation(self, mi_init, mi_fin, mi_step):
        """Trains the network and calculates the MI every mi_step epochs"""
        for iMI in range(mi_init, mi_fin):
            hist = self.model.fit(self.train_confs, self.train_labels, 
                                  batch_size=self.batch_size, epochs=mi_step, 
                 validation_data=(self.test_confs, self.test_labels), verbose=0)
            # Metrics
            self.metrics.append(hist.history)
    
            # Calculate MI
            net_out = nf.get_all_layers(self.get_layer, self.total_confs)
            for iT in range(len(net_out)):
                self.MI_temp[iT] = self.mi_obj.mut_info(net_out[iT]))
            self.MI.append(self.MI_temp)  
            
            # Print messages if given b
            if self.print_step != None:
                if (iMI - mi_init) % self.print_step == 0:
                    print('%d / %d done!'%(iMI+1, mi_fin))
        
    def train_with_schedule(self, schedule):
        """Schedule is a 2D array where the first column is the number of epochs and
        the second column is the mi_step for these epochs."""
        # Update total number of epochs
        n_sch = schedule.shape[1]
        self.n_epoch += np.sum(schedule, axis=0)[0]
        miv = np.int(schedule[:, 0] / schedule[:, 1])
        
        self._train_with_mi_calculation(0, miv[0], schedule[0,1])
        for i in range(1,n_sch):
            self._train_with_mi_calculation(miv[i-1], miv[i], schedule[i,1])           
        
    def train_without_schedule(self, mi_step=1, n_epoch):
        """Give a constant mi_step for training"""
        mi_fin = int(n_epoch / mi_step)
        # Update the total number of epochs
        self.n_epoch += n_epoch
        self._train_with_mi_calculation(0, mi_fin, mi_step)

    def save_data(self):
        """Saves MI and metrics data"""
        self.assign_name()
        nf.create_data_directories()
        np.save('Metrics/' + self.network_name +'_MIdata.npy', np.array(self.MI))
        np.save('MIvalues/'+ self.network_name +'_metrics.npy', nf.get_metrics(self.metrics))
        
    def network_checkpoint(self):
        """Saves network architecture, weights and biases"""
        self.assign_name()
        nf.create_directory('Networks/')
        ndir = 'Networks/' + self.network_name
        nf.create_directory(ndir)
        self.model.save(ndir+'.h5')
        
        
