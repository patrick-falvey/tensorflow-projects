
from time import time
import tensorflow as tf
mnist = tf.keras.datasets.mnist
# mnist = test.load_data()
from tensorflow.keras.callbacks import TensorBoard

NAME = "Digits {}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Model Definition
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross Entropy Definition
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# Train the model
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)

# Launch interactive Session
sess = tf.InteractiveSession()

# Create Operation
tf.global_variables_initializer().run()

# Run the training step 1000 times
for _ in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



