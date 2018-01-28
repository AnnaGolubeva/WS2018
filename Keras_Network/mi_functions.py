# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 18:29:03 2018

@author: Matt
"""

### with some modifications from the original ###
import numpy as np

class MI_class(object):
    def __init__(self, data_folder, n_bins=31):
        
        # Initialize Mutual Information Class for given dataset
        # mut_info method will be called with layer output array:
        #     t = (Nx by n_t) array, where Nx is the number of datasets, and n_t is the number of neurons in the layer
        
        # Define bin size
        self.n_bins = n_bins
        
        # Define data choice
        if data_folder == 'toy_dataset':
            self.X = np.load('toy_dataset/labels64.npy')
            self.theta = 34
        elif data_folder == 'toy_tishby':
            self.X = np.sum(np.load('toy_tishby/configs.npy'), axis=1)
            self.theta = 6

        # Obtain constant p_xy distribution from dataset and calculate: p(x) and size of probability space: NX
        self.p_xy = self.prob_joint_X_binaryY()
        
        return
        
    def mut_info(self,t):
        # Estimate Mutual Information of between Random Variables (X,Y,T):    I(X,T) and I(Y,T)
        
        # Probability of p(t(x)) and delta(t(x),t(x'))
        p_tx, delta_tx = self.prob_t_x(t)      
        
        # Calculate Mutual Information of I(T,Y)
        # p_xy: (Ny(=2 for binary) by Nx) array,    p_tx = (Nx by 1) array,   delta_tx = (Nx by Nx) array,  p_x = (Nx by 1) array
        
        ###### GET nan in log2 part of I_TY: due to p_xy values rounded to 0's and 1's for unknown reason ###########
        #I_TY = np.nansum(self.p_xy * np.log2(np.dot(self.p_xy, delta_tx)))
        #I_TY += -np.nansum(self.p_xy * np.log2(np.sum(self.p_xy, 1)[:,np.newaxis] * p_tx))
        I_TY = np.nansum(self.p_xy*(np.log2(np.dot(self.p_xy,delta_tx)/np.sum(
                self.p_xy,1)[:,np.newaxis]/p_tx)))
    
        I_TX = -np.dot(self.p_x, np.log2(p_tx))


        return np.array([I_TX, I_TY])
             
    def prob_joint_X_binaryY(self):
    
        def py_x(u,gamma=30.5,th=34):
            #return 1.0/(1.0 + np.exp(-gamma*(np.sum(u, axis=1) - 6)))
            return 1.0/(1.0 + np.exp(-gamma*(u-th)))

        # Import Original X Data and calculate size of Probability Space NX
        self.NX = len(self.X)
        
        # Calculate p(x)
        self.p_x = np.ones(self.NX)*1/self.NX
        
        ################ when printed, shows py_x values rounded to 0's and 1's for unknown reason ###########
        pyx = py_x(self.X, th = self.theta) 
        #print('pyx sig: ',pyx)
        
        return np.array([(1-pyx)*self.p_x, pyx*self.p_x])

    def prob_t_x(self, t): # Thanks Lauren!
        # """Takes the layer's output t(x) and a number of bins
        #  Returns a probability p(t(x)) as a vector and a matrix for KroneckerDelta(t(x), t(x'))"""

        # Define bins
        bins = np.linspace(-1, 1, self.n_bins)
        
        # Count number of appearance of each vector
        _, indices, counts= np.unique(np.digitize(t, bins), 
                                return_inverse=True, return_counts=True, axis=0)
        # Create delta matrix from indices
        delta = (np.array([indices,] * len(indices)).T == indices).astype(np.int)
        
        # Return p(t_x), delta
        return counts[indices]/self.NX, delta