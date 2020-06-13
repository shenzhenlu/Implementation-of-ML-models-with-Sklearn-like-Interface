import numpy as np
import pandas as pd
from sklearn.utils import shuffle


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) * np.sqrt(2 / M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return  1 / (1+np.exp(-A))


def softmax(A):
    return np.exp(A) / np.exp(A).sum(axis=1, keepdims=True)


def sigmoid_cost(T, predictions):
    return -(T*np.log(predictions) + (1-T)*np.log(1-predictions)).sum()


def softmax_cost(T, predictions):
    return -(T*np.log(predictions)).sum()


def softmax_cost2(Y, predictions):
    """Same as cost(), it just uses raw targets to index y to aviod 
    multiplication by a large indicator maxtrix."""
    return -np.log(predictions[np.arange(len(predictions)), Y]).sum()


def error_rate(target, predictions):
    return np.mean(target != predictions)


def y2indicator(Y):
    N = len(Y)
    K = len(set(Y))
    T = np.zeros((N, K))
    for i in range(N):
        T[i, Y[i]] = 1
    return T
    

def get_data(balance_ones=True, N_test=1000):
    y = []
    X = []
    is_first_line = True
    for line in open('../Data/fer2013.csv'):
        if is_first_line:
            is_first_line = False
        else:
            row = line.split(',')
            y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])
            
    X, y = np.array(X) / 255.0, np.array(y)
    
    # Shuffle and split
    np.random.seed(1)
    X, y = shuffle(X, y)
    X_train, y_train = X[:-N_test], y[:-N_test]
    X_val, y_val = X[-N_test:], y[-N_test:]
    
    if balance_ones:
        # Blance the 1 class
        X0, y0 = X_train[y_train != 1, :], y_train[y_train != 1]
        X1 = X_train[y_train == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        
        X_train = np.concatenate((X0, X1), axis=0)
        y_train = np.concatenate((y0, [1]*len(X1)))
        
    return X_train, y_train, X_val, y_val 


def get_binary_data(balance_ones=True, N_test=1000):
    y = []
    X = []
    is_first_line = True
    for line in open('../Data/fer2013.csv'):
        if is_first_line:
            is_first_line = False
        else:
            row = line.split(',')
            label = int(row[0])
            if label == 0 or label == 1:
                y.append(label)
                X.append([int(p) for p in row[1].split()])
                
    X, y = np.array(X) / 255.0, np.array(y)
    
    # Shuffle and split
    np.random.seed(1)
    X, y = shuffle(X, y)
    X_train, y_train = X[:-N_test], y[:-N_test]
    X_val, y_val = X[-N_test:], y[-N_test:]
    
    if balance_ones:
        # Blance the 1 class
        X0, y0 = X_train[y_train != 1, :], y_train[y_train != 1]
        X1 = X_train[y_train == 1, :]
        X1 = np.repeat(X1, 9, axis=0)
        
        X_train = np.concatenate((X0, X1), axis=0)
        y_train = np.concatenate((y0, [1]*len(X1)))
        
    return X_train, y_train, X_val, y_val 