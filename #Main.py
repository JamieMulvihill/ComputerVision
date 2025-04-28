#Main.py

import tensorflow as tf
import keras as ks
import urllib.request
import zipfile
import tensorflow_datasets as tfds

from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from keras import layers
from keras._tf_keras.keras.optimizers import RMSprop

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.95):
            print('\nReached  95% accuracy so calling callback')
            self.model.stop_training = True

training_dir = 'horse-or-human/train/'
validation_dir = 'horse-or-human/validation/'

train_datagen = ImageDataGenerator(rescale=1./255,
 rotation_range=40,
 width_shift_range=0.2,
 height_shift_range=0.2,
 shear_range=0.2,
 zoom_range=0.2,
 horizontal_flip=True,
 fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)

test_data = tfds.load('horses_or_humans', split='test', as_supervised=True)
IMG_SIZE = 300
def preprocess_test_image(image, label):
    resized_image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    scaled_image = tf.cast(resized_image, tf.float32) / 255.0
    return scaled_image, label

test_dataset = test_data.map(preprocess_test_image).batch(32).prefetch(tf.data.AUTOTUNE)


callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist

NumPixelX = 300
NumPixelY = 300
NumColorChannels = 3

model = Sequential([
    layers.Conv2D(16, (3,3), activation='relu',
      input_shape=(NumPixelX, NumPixelY, NumColorChannels)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=0.001),
              metrics=['accuracy'])

#callbacks=[callbacks]
history = model.fit(
    train_generator, 
    epochs=15, 
    validation_data=validation_generator
    )
print("Done")
print("\nEvaluating on Test Data...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")