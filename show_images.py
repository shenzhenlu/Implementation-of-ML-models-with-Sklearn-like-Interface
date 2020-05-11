#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:13:13 2020

@author: lu
"""

import numpy as np
import matplotlib.pyplot as plt

from util import get_data

labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def main():
    X, Y = get_data(balance_ones=False)
    
    while True:
        for i in range(7):
            x, y = X[Y==i], Y[Y==i]
            N = len(y)
            j = np.random.choice(N)
            
            plt.imshow(x[j].reshape(48, 48), cmap='gray')
            plt.title(labels[y[j]])
            plt.show()
            
        prompt = input('Quit? Enter y:\n')
        if prompt == 'y':
            break


if __name__ == '__main__':
    main()        