# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 11:14:25 2022

@author: pelka
"""

# Redes Neuronales Convulsionales
# Part 1 - Make CNN model
import tensorflow as tf

# Import libraries
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Init CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(filters=32, kernel_size= (3, 3),
                      input_shape=(64, 64, 3), activation="relu"))

# Step 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Second layer of convulution and max pooling
classifier.add(Conv2D(filters=32, kernel_size= (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

#Third layer with 32 filters and max pooling
classifier.add(Conv2D(filters=64, kernel_size= (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full Conection
classifier.add(Dense(units = 128, activation="relu"))
classifier.add(Dense(units = 128, activation="relu"))
classifier.add(Dense(units = 1, activation="sigmoid"))

# Compile CNN
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Part 2 - Adjust CNN for training images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('./dataset/training_set',
                                                target_size=(64, 64),
                                                batch_size=32,
                                                class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('./dataset/test_set',
                                               target_size=(64, 64),
                                               batch_size=32,
                                               class_mode='binary')

with tf.device("/GPU:0"):
    classifier.fit(training_dataset,
                steps_per_epoch=(8000)/32,
                epochs=25,
                validation_data=testing_dataset,
                validation_steps=(2000)/32)
