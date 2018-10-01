import pickle
from time import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
mnist = tf.keras.datasets.mnist
from tensorflow.keras.callbacks import TensorBoard


dense_layers = 2
layer_sizes = [64]
activation_types = ['relu', 'sigmoid']

for dense_layer in range(1, dense_layers):
    for size in layer_sizes:
        for activation in activation_types:

            # Name Model and setup Tensorboard
            NAME = "Activation-{} Dense-{} Layer Size-{} {}".format(activation, dense_layer, size, int(time()))
            tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

            # Unpack Data
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            x_train, x_test = x_train / 255.0, x_test / 255.0

            # Define Model
            model = Sequential()

            for _ in range(1, dense_layer+1):
                model.add(Conv2D(size, (3, 3), input_shape=x_train.shape[1:]))
                model.add(Activation('relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))

            model.add(Flatten())
            model.add(Dense(10, activation='softmax'))

            model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            model.fit(
                x_train,
                y_train,
                epochs=3,
                callbacks=[tensorboard])

            model.evaluate(x_test, y_test)



