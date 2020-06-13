import numpy as np
import matplotlib.pyplot as plt
from util import get_binary_data, sigmoid, sigmoid_cost, error_rate, relu

class ANN(object):
    def __init__(self, M):
        self.M = M
        
    def fit(self, X_train, Y_train, X_val, Y_val, learning_rate=1e-6, lambda_=1.0, epochs=10000, show_fig=False):        
        N, D = X_train.shape
        self.W1 = np.random.randn(D, self.M) * np.sqrt(1 / D)
        self.b1 = np.zeros(self.M)
        self.W2 = np.random.randn(self.M) * np.sqrt(1 / self.M)
        self.b2 = 0
        
        costs = []
        best_val_error = 1
        for i in range(epochs):
            # Forward Propagation
            Y_train_pred, Z = self.forward(X_train)
            
            # Gradient Descent step
            delta2 = Y_train_pred - Y_train
            self.W2 -= learning_rate * (Z.T.dot(delta2) + reg*self.W2)
            self.b2 -= learning_rate * (delta2.sum(axis=0) + reg*self.b2)
            
            delta1 = delta2.dot(self.W2.T) * (Z > 0)
            self.W1 -= learning_rate * (X_train.T.dot(delta1) + reg*self.W1)
            self.b1 -= learning_rate * (delta1.sum(axis=0) + reg*self.b1)
            
            if i % 50 == 0:
                Y_val_pred, _ = self.forward(X_val)
                c = sigmoid_cost(Y_val, Y_val_pred)
                costs.append(c)
                e = error_rate(Y_val, np.round(Y_val_pred))
                print("Epoch:", i, "Cost:", c, "Error rate:", e)
                if e < best_val_error:
                    best_val_error = e    
        print("Best validation error:", best_val_error)
            
        if show_fig:
            plt.plot(costs)
            plt.show()
            
    def forward(self, X_train):
        Z = relu(X_train.dot(self.W1) + self.b1)
        return sigmoid(Z.dot(self.W2) + self.b2), Z


def main():
    X_train, Y_train = get_binary_data()
    
    model = ANN(100)
    model.fit(X_train, Y_train, show_fig=True)


if __name__ == '__main__':
    main()