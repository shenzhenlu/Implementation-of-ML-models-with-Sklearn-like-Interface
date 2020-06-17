import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from util import get_image_data, error_rate, init_weight_and_bias, y2indicator, init_filter
from ann_softmax_tf import HiddenLayer


class ConvPoolLayer(object):
    def __init__(self, mi, mo, fw=5, fh=5, poolsz=(2, 2)):
        filt_sz = (fw, fh, mi, mo)
        W0 = init_filter(filt_sz)
        self.W = tf.Variable(W0)
        b0 = np.zeros(mo, dtype=np.float32)
        self.b = tf.Variable(b0)
        self.params = [self.W, self.b]
        self.poolsz = poolsz
        
    def forward(self, X):
        conv_out = tf.nn.conv2d(X, self.W, strides=[1, 1, 1, 1], padding='SAME')
        conv_out = tf.nn.bias_add(conv_out, self.b)
        p1, p2 = self.poolsz
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize=[1, p1, p2, 1],
            strides=[1, p1, p2, 1],
            padding='SAME'
            )
        return tf.nn.relu(pool_out)


class CNN(object):
    def __init__(self, convpool_layer_sizes, hidden_layer_sizes):
        self.convpool_layer_sizes = convpool_layer_sizes
        self.hidden_layer_sizes = hidden_layer_sizes

    def fit(self, X_train, labels_train, X_val, labels_val, learning_rate=1e-4, mu=0.9,
            decay=0.99, lambda_=1e-3, epochs=5, batch_sz=200, show_fig=False):
        K = len(set(labels_train))
        
        # Correct datatype
        X_train, X_val = X_train.astype(np.float32), X_val.astype(np.float32)
        Y_train, Y_val = y2indicator(labels_train).astype(np.float32), y2indicator(labels_val).astype(np.float32)
        
        # Initialize convpool layers
        N, width, height, c = X_train.shape
        mi = c
        outw = width
        outh = height
        self.convpool_layers = []
        for mo, fw, fh in self.convpool_layer_sizes:
            cp = ConvPoolLayer(mi, mo, fw, fh)
            self.convpool_layers.append(cp)
            outw = outw // 2
            outh = outh // 2
            mi = mo
            
        # Initialize hidden layers
        self.hidden_layers = []
        M1 = mi * outw * outh
        count = 0
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2
            count += 1
        
        # Initialize Output Layer
        W, b = init_weight_and_bias(M1, K)
        self.W = tf.Variable(W)
        self.b = tf.Variable(b)
        
        # Collect params for later use
        self.params = [self.W, self.b]
        for cp in self.convpool_layers:
            self.params += cp.params
        for h in self.hidden_layers:
            self.params += h.params
        
        # Set up tensorflow functions and variables
        tf_X = tf.placeholder(tf.float32, shape=(None, width, height, c), name='X')
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
        for cp in self.convpool_layers:
            Z = cp.forward(Z)
        Z_shape = Z.get_shape().as_list()
        Z = tf.reshape(Z, [-1, np.prod(Z_shape[1:])])
        for h in self.hidden_layers:
            Z = h.forward(Z)
        return tf.matmul(Z, self.W) + self.b
        
    def predict(self, X):
        logits = self.forward(X)
        return tf.argmax(logits, 1)
        
    
def main():
    X_train, labels_train, X_val, labels_val = get_image_data()

    model = CNN(convpool_layer_sizes=[(20, 5, 5), (20, 5, 5)],
                hidden_layer_sizes=[300, 100])
    model.fit(X_train, labels_train, X_val, labels_val, show_fig=True)


if __name__ == '__main__':
    main()
    