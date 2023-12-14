import os
import matplotlib.pyplot as plt 
import numpy as np
import random, shutil
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model

# Function for generating data batches
def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True, batch_size=1, target_size=(24,24), class_mode='categorical'):
    return gen.flow_from_directory(dir, batch_size=batch_size, shuffle=shuffle, color_mode='grayscale', class_mode=class_mode, target_size=target_size)

Batch_size = 32
Img_shape = (24, 24)

# Generating training and validation batches for eyes and mouth classification
train_batch = generator('data/train', shuffle=True, batch_size=Batch_size, target_size=Img_shape)
valid_batch = generator('data/valid', shuffle=True, batch_size=Batch_size, target_size=Img_shape)

# Steps per epoch for training and validation
Steps = len(train_batch.classes) // Batch_size
Valid_steps = len(valid_batch.classes) // Batch_size
print(Steps, Valid_steps)

# Model architecture for eyes and mouth classification
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24, 24, 1)),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1, 1)),
    
    Dropout(0.25),
    Flatten(),
    
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(4, activation='softmax')  # Output layer with 4 neurons for eyes and mouth (2 classes each)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model for eyes and mouth classification
model.fit_generator(train_batch, validation_data=valid_batch, epochs=15, steps_per_epoch=Steps, validation_steps=Valid_steps)

# Saving the trained model
model.save('models/cnn_eyes_mouth.h5', overwrite=True)
