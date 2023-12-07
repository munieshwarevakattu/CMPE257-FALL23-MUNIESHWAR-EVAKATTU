import os
import matplotlib.pyplot as plt 
import numpy as np
import random,shutil
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

Batch_size= 32
Img_shape=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=Batch_size,target_size=Img_shape)
valid_batch= generator('data/valid',shuffle=True, batch_size=Batch_size,target_size=Img_shape)
Steps= len(train_batch.classes)//Batch_size
Valid_steps = len(valid_batch.classes)//Batch_size
print(Steps,Valid_steps)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3, 3), activation='relu'),#32 convolution filters used each of size 3x3 with relu as activation
    MaxPooling2D(pool_size=(1,1)),

    
    Dropout(0.25),#randomly turn neurons on and off to improve convergence and aviod overfittng
    Flatten(),# flatterns the data to chage the dimmesions

    Dense(128, activation='relu'),# pushing them to real densed network
    Dropout(0.5),
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

model.save('models/cnnimg.h5', overwrite=True)