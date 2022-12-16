# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 10:34:48 2022

@author: HP
"""

import Perceptrons_utils as pu
import pickle 
import numpy as np
import torch as t
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

N=4
L=4
Nsamples,Epochs,batchsize=100,120,10
dts = [0.1]

#pu.build_and_store_data_sets(N, Nsamples)

#Training: 
file = 'Data_N'+str(N)+'_Model_Heis_Nsamps'+str(Nsamples)+'_.pt'

with open(file, 'rb') as infile: 
     training_data=pickle.load(infile)
   
Xtrain,Xtest = training_data['X'], training_data['Xtest']
ytrain,ytest = training_data['y'], training_data['ytest']

witness = pu.Witness(N,L,dts)

#Training
losses = witness.train(Xtrain, ytrain, Epochs, batchsize)

plt.figure()
plt.plot(range(len(losses)), losses, linewidth=4, color='blue')
plt.xlabel('Epoch', fontsize=16)
plt.ylabel('Mean-square loss', fontsize=16)
plt.show()

#Accuracies for the Heisenberg test set
TP_Heis, TN_Heis, FP_Heis, FN_Heis, outputs = witness.accuracy(Xtest,ytest)

#Testing on other modes
Nsamples=100
models=['XY', 'Ising','XX-Z', 'XYZ']
TPs = t.zeros(len(models))
TNs = t.zeros(len(models))
FPs = t.zeros(len(models))
FNs = t.zeros(len(models))
test_outputs = []

for m in range(len(models)):
    file = 'Data_N'+str(N)+'_Model_'+models[m]+'_Nsamps'+str(Nsamples)+'_.pt'
    with open(file, 'rb') as infile:
        training_data=pickle.load(infile)
    X,y = training_data['X'], training_data['y']
    a,b,c,d,e = witness.accuracy(X,y)
    TPs[m],TNs[m],FPs[m],FNs[m]=a,b,c,d
    test_outputs.append(e)
    
plt.figure()
plt.scatter(range(len(models)), TPs, color='red', marker='x', label='True Positives')
plt.scatter(range(len(models)), TNs, color='blue', marker='o',label='True Negatives')
plt.xticks(ticks=range(len(models)), labels=models, fontsize=14)
plt.ylabel('Percentage', fontsize=16)
plt.legend(fontsize=14)
plt.show()
