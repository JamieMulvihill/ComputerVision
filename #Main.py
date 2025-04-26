#Main.py

import tensorflow as tf
import keras as ks

from keras import Sequential
from keras import layers

(training_images, training_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()

print(training_images.shape)
print(training_labels.shape)
print(test_images.shape)
print(test_labels.shape)

training_images = training_images/255
test_images = test_images/255

model = Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=10)

print("Done")

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])