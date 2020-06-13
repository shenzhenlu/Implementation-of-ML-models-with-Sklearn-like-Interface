import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from util import get_data, softmax, softmax_cost2, y2indicator, error_rate

class LogisticModel():
    def __init__(self):
        pass
    
    def fit(self, X_train, labels_train, X_val, labels_val, learning_rate=5e-7, lambda_=1e0, epochs=5000, show_fig=False):
        N, D = X_train.shape
        K = len(set(labels_train))
        Y_train = y2indicator(labels_train)
        self.W = np.random.randn(D, K) * np.sqrt(1 / D)
        self.b = np.zeros(K)
        
        costs = []
        best_val_error = 1
        for i in range(epochs):
            # Forward propagation 
            Y_train_pred = self.forward(X_train)
            
            # Gradient descent
            self.W -= learning_rate * (X_train.T.dot(Y_train_pred-Y_train) + lambda_*self.W)
            self.b -= learning_rate * ((Y_train_pred-Y_train).sum(axis=0) + lambda_*self.b)
        
            if i % 50 == 0:
                Y_val_pred = self.forward(X_val)
                c = softmax_cost2(labels_val, Y_val_pred)
                costs.append(c)
                e = error_rate(labels_val, np.argmax(Y_val_pred, axis=1))
                print("Epoch:", i, "Cost:", c, "Error rate", e)
                if e < best_val_error:
                    best_val_error = e
        print("Best validation error", best_val_error)
        
        if show_fig:
            plt.plot(costs)
            plt.show()
        
    def forward(self, X):
        return softmax(X.dot(self.W) + self.b)
    
    
def main():
    X_train, labels_train, X_val, labels_val = get_data()
    
    model = LogisticModel()
    model.fit(X_train, labels_train, X_val, labels_val, show_fig=True)


if __name__ == '__main__':
    main()
    
