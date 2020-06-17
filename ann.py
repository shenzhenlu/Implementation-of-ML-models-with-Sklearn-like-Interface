import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import get_data, softmax, softmax_cost2, error_rate, relu, y2indicator

class ANN(object):
    def __init__(self, M):
        self.M = M
        
    def fit(self, X_train, labels_train, X_val, labels_val, 
            learning_rate=5e-7, lambda_=1e0, epochs=5000, show_fig=False):
        N, D = X_train.shape
        K = len(set(labels_train))
        Y_train = y2indicator(labels_train)
        self.W1 = np.random.randn(D, self.M) * np.sqrt(2 / (D + self.M))
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M, K) * np.sqrt(2 / (self.M + K))
        self.b2 = np.zeros(K)
        
        costs = []
        best_val_error = 1
        for i in range(epochs):
            # Forward Propagation
            Y_train_pred, Z = self.forward(X_train)
            
            # Gradient Descent step
            delta2 = Y_train_pred - Y_train
            self.W2 -= learning_rate * (Z.T.dot(delta2) + lambda_*self.W2)
            self.b2 -= learning_rate * (delta2.sum(axis=0) + lambda_*self.b2)
            
            #delta1 = np.outer(delta2, self.W2) * (Z > 0)
            delta1 = delta2.dot(self.W2.T) * (1 - Z*Z)
            self.W1 -= learning_rate * (X_train.T.dot(delta1) + lambda_*self.W1)
            self.b1 -= learning_rate * (delta1.sum(axis=0) + lambda_*self.b1)
            
            if i % 50 == 0:
                Y_val_pred, _ = self.forward(X_val)
                c = softmax_cost2(labels_val, Y_val_pred)
                costs.append(c)
                e = error_rate(labels_val, np.argmax(Y_val_pred, axis=1))
                print("i:", i, "cost:", c, "error:", e)
                if e < best_val_error:
                    best_val_error = e    
        print("best_val_error:", best_val_error)
            
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def forward(self, X):
        Z = np.tanh(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2), Z
    
    
def main():
    X_train, labels_train, X_val, labels_val = get_data()

    model = ANN(100)
    model.fit(X_train, labels_train, X_val, labels_val, show_fig=True)


if __name__ == '__main__':
    main()