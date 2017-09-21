import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
'''
basic forward feed neural network designed to train the mnist data set based on
lecture series be sentdex on youtube.  This has been generalized to take
dynamically sized layers
'''

class TF_NN:

    def __init__(self, node_counts, epochs=10):
        # Import MNIST data
        self.mnist = input_data.read_data_sets("/tmp/MNIST_data", one_hot=True)

        self.epochs = epochs

        # count of the nodes per layer starting with the input layer, ending
        # with output
        self.node_counts = node_counts

        self.n_classes = node_counts[-1]
        self.batch_size = 100

        # height x width
        self.x = tf.placeholder(tf.float32, [None, node_counts[0]])
        self.y = tf.placeholder(tf.float32)


    def neural_network_model(self, data):
        layers = []
        a = []
        z = []

        a.append(data)

        for i, _ in enumerate(self.node_counts):
            # skip first index
            # TODO: find a more elegant solution
            if i == 0:
                continue

            weight_dict = {'weights': tf.Variable(tf.random_normal([self.node_counts[i - 1], self.node_counts[i]])), 'biases': tf.Variable(tf.random_normal([self.node_counts[i]]))}
            layers.append(weight_dict)

            z.append(tf.add(tf.matmul(a[-1], layers[i - 1]['weights']), layers[i - 1]['biases']))
            a.append(tf.nn.relu(z[-1]))

        return z[-1]

    def train_neural_network(self, x):
        prediction = self.neural_network_model(x)
        cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=self.y))
        
        # learning_rate can be specified
        optimizer = tf.train.AdamOptimizer(learning_rate = .003).minimize(cost)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(self.epochs):
                epoch_loss = 0
                for _ in range(int(self.mnist.train.num_examples/self.batch_size)):
                    x_batch, y_batch = self.mnist.train.next_batch(self.batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict = {self.x: x_batch, self.y: y_batch})
                    epoch_loss += c
                print("Epoch", epoch + 1, "completed out of", self.epochs, "loss", epoch_loss)
                
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(self.y, 1))
            
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({self.x:self.mnist.test.images, self.y:self.mnist.test.labels}))


if __name__ == "__main__":
    nn = TF_NN([784, 35, 10])
    nn.train_neural_network(nn.x)

