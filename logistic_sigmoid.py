#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 12:12:32 2020

@author: lu
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from util import get_binary_data, sigmoid, sigmoid_cost, error_rate

class LogisticModel():
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate=10e-6, lambda_=0.0, epochs=20000, show_fig=False):
        np.random.seed(seed=87)
        
        X, Y = shuffle(X, Y)
        X_val, Y_val = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]
        
        N, D = X.shape
        self.W = np.random.randn(D) / np.sqrt(D)
        self.b = 0
        
        costs = []
        best_val_error = 1
        for i in range(epochs):
            # Forward propagation 
            y_pred = self.forward(X)
            
            # Gradient descent
            self.W -= learning_rate * (X.T.dot(y_pred-Y) + lambda_*self.W)
            self.b -= learning_rate * ((y_pred-Y).sum() + lambda_*self.b)
        
            if i % 20 == 0:
                Y_val_pred = self.forward(X_val)
                c = sigmoid_cost(Y_val, Y_val_pred)
                costs.append(c)
                e = error_rate(Y_val, np.round(Y_val_pred))
                print("i:", i, "cost:", c, "error", e)
                if e < best_val_error:
                    best_val_error = e
        print("best validation error", best_val_error)
        
        if show_fig:
            plt.plot(costs)
            plt.show()
        
    def forward(self, X):
        return sigmoid(X.dot(self.W) + self.b)
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.round(y_pred)
    
    def score(self, X, Y):
        prediction = self.predict(X)
        return 1 - error_rate(Y, prediction)
    
    
def main():
    X, Y = get_binary_data()
    
    X0 = X[Y==0, :]
    X1 = X[Y==1, :]
    X1 = np.repeat(X1, 9, axis=0)
    X = np.concatenate([X0, X1])
    
    Y0 = Y[Y==0]
    Y1 = Y[Y==1]
    Y1 = np.repeat(Y1, 9, axis=0)
    Y = np.concatenate([Y0, Y1])
    
    model = LogisticModel()
    model.fit(X, Y, show_fig=True)


if __name__ == '__main__':
    main()
    
