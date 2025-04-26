#Main.py

import tensorflow as tf
import keras as ks

from keras import Sequential
from keras import layers

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print('\nReached  95% accuracy so calling callback')
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

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

model.fit(training_images, training_labels, epochs=50, callbacks=[callbacks])

print("Done")

model.evaluate(test_images, test_labels)

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])