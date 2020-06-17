import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from util import get_data, y2indicator, error_rate, init_weight_and_bias
from sklearn.utils import shuffle

class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weight_and_bias(M1, M2)
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)
        self.params = [self.W, self.b]
        
    def forward(self, X):
        return tf.nn.relu(tf.matmul(X, self.W) + self.b)


class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
        
    def fit(self, X_train, labels_train, X_val, labels_val, learning_rate=1e-4, mu=0.9, 
            decay=0.99, lambda_=1e-3, epochs=30, batch_sz=200, show_fig=False):
        K = len(set(labels_train))
        
        # Correct datatype
        X_train, X_val = X_train.astype(np.float32), X_val.astype(np.float32)
        Y_train, Y_val = y2indicator(labels_train).astype(np.float32), y2indicator(labels_val).astype(np.float32)
        
        # Initialize hidden layers
        N, D = X_train.shape
        self.hidden_layers = []
        M1 = D
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)
        
        # Collect params
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
            
        tf_X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        tf_Y = tf.placeholder(tf.float32, shape=(None, K), name='Y')
        logits = self.forward(tf_X)
        
        reg_cost = lambda_ * sum([tf.nn.l2_loss(p) for p in self.params])
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, 
                labels=tf_Y
            )
        ) + reg_cost
        train_op = tf.train.RMSPropOptimizer(learning_rate, decay=decay, momentum=mu).minimize(cost)
        
        predict_op = self.predict(tf_X)
        
        n_batches = N // batch_sz
        costs = []
        best_val_error = 1
        init = tf.global_variables_initializer()
        with tf.Session() as session:
            session.run(init)
            for i in range(epochs):
                X_train, Y_train = shuffle(X_train, Y_train)
                for j in range(n_batches):
                    Xbatch = X_train[j*batch_sz:(j*batch_sz+batch_sz)]
                    Ybatch = Y_train[j*batch_sz:(j*batch_sz+batch_sz)]
                    
                    session.run(train_op, feed_dict={tf_X: Xbatch, tf_Y: Ybatch})
                    
                    if j % 20 == 0:
                        c = session.run(cost, feed_dict={tf_X: X_val, tf_Y: Y_val})
                        costs.append(c)
                        
                        labels_val_pred = session.run(predict_op, feed_dict={tf_X: X_val})
                        e = error_rate(labels_val, labels_val_pred)
                        print("i:", i, "j:", j, '/', n_batches, "cost:", c, "error_rate:", e)
                        if e < best_val_error:
                            best_val_error = e    
            print("best_val_error:", best_val_error)
            
        if show_fig:
            plt.plot(costs)
            plt.show()
        
    def forward(self, X):
        Z = X
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b
        
    def predict(self, X):
        logits = self.forward(X)
        return tf.argmax(logits, 1)
        
    
def main():
    X_train, labels_train, X_val, labels_val = get_data()

    model = ANN([200, 100, 50])
    model.fit(X_train, labels_train, X_val, labels_val, show_fig=True)


if __name__ == '__main__':
    main()
    