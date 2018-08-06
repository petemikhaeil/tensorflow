import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.set_printoptions(threshold=np.nan)
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

h1 = 200
h2 = 100
h3 = 60
h4 = 30

# None represents batch size (think of none as a mark), and then width and height and then depth (if rbg then would be 3)
input = tf.placeholder(tf.float32, [None, 784])
w1 = tf.Variable(tf.truncated_normal([784, h1], stddev=0.1))
b1 = tf.Variable(tf.zeros([h1]))

w2 = tf.Variable(tf.truncated_normal([h1, h2], stddev=0.1))
b2 = tf.Variable(tf.zeros([h2]))

w3 = tf.Variable(tf.truncated_normal([h2, h3], stddev=0.1))
b3 = tf.Variable(tf.zeros([h3]))

w4 = tf.Variable(tf.truncated_normal([h3, h4], stddev=0.1))
b4 = tf.Variable(tf.zeros([h4]))

w5 = tf.Variable(tf.truncated_normal([h4, 10], stddev=0.1))
b5 = tf.Variable(tf.zeros([10]))

# y is the output of the model
hidden_1 = tf.nn.relu(tf.matmul(input, w1) + b1)
# hidden_1 = tf.nn.dropout(hidden_1, 0.8),
hidden_2 = tf.nn.relu(tf.matmul(hidden_1, w2) + b2)
# hidden_2 = tf.nn.dropout(hidden_2, 0.85)
hidden_3 = tf.nn.relu(tf.matmul(hidden_2, w3) + b3)
# hidden_3 = tf.nn.dropout(hidden_3, 0.9)
hidden_4 = tf.nn.relu(tf.matmul(hidden_3, w4) + b4)
# hidden_4 = tf.nn.dropout(hidden_4, 0.95)
output_layer = tf.nn.softmax(tf.matmul(hidden_4, w5) + b5)

# Will be true distribution
y_ = tf.placeholder(tf.float32, [None, 10])

loss = -tf.reduce_sum(y_ * tf.log(output_layer))

is_correct = tf.equal(tf.argmax(output_layer, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.AdamOptimizer()
train_step = optimizer.minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
accuracies = []
losses = []
for i in range(5000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_data = {input: batch_x, y_: batch_y}
    sess.run(train_step, feed_dict=train_data)
    a, c = sess.run([accuracy, loss], feed_dict=train_data)
    test_data = {input: mnist.test.images, y_: mnist.test.labels}
    a, c = sess.run([accuracy, loss], feed_dict=test_data)
    print(i)
    accuracies.append(a)
    losses.append(c)

plt.subplot(2, 1, 1)
plt.plot(accuracies)
plt.title('Accuracies on Test Set')
plt.ylabel('Accuracy Percentage')
plt.subplot(2, 1, 2)
plt.plot(losses)
plt.title('Loss on Test Set')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()
print(accuracies[-1])
