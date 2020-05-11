#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:56:21 2020

@author: lu
"""

import numpy as np
import pandas as pd


def get_data(balance_ones=True):
    Y = []
    X = []
    is_first_line = True
    for line in open('fer2013.csv'):
        if is_first_line:
            is_first_line = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
            
    X, Y = np.array(X) / 255.0, np.array(Y)
    
    if balance_ones:
        # blance the 1 class
        X0, Y0 = X[Y != 1, :], Y[Y != 1]
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, np.ones((len(X1), 1))))
        
    return X, Y
        
def sigmoid(A):
    return  1 / (1+np.exp(-A))

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

def error_rate(targets, predictions):
    return np.mean(targets != predictions)

def get_binary_data():
    Y = []
    X = []
    is_first_line = True
    for line in open('fer2013.csv'):
        if is_first_line:
            is_first_line = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
                
    return np.array(X) / 255.0, np.array(Y)