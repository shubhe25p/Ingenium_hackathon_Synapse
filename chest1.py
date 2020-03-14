import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import LeakyReLU
from keras.layers import Flatten
from keras.layers import Dense
classifier=Sequential()
classifier.add(Convolution2D(16,3,3, input_shape=(224,224,3)))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,3,3))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(16,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(128,3,3))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(16,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(128,3,3))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(32,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(256,3,3))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(32,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(256,3,3))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Convolution2D(64,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(512,3,3))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(64,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(512,3,3))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(128,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(Convolution2D(10,1,1))
classifier.add(LeakyReLU(alpha=0.1))
classifier.add(AveragePooling2D(pool_size=(6,6)))
classifier.add(Flatten())

classifier.add(Dense(output_dim=2,activation='sigmoid'))

classifier.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2
                                 )
train_datagen=ImageDataGenerator(rescale=1./255)
training_set=train_datagen.flow_from_directory('C:/Users/shubh/Desktop/ingeniumhackathon_Synapse/chest_xray/train', target_size=(224,224) ,batch_size=100,class_mode='categorical')
test_set=train_datagen.flow_from_directory('C:/Users/shubh/Desktop/ingeniumhackathon_Synapse/chest_xray/test', target_size=(224,244),batch_size=100,class_mode='categorical')
classifier.fit_generator(training_set,
                         samples_per_epoch=8000,
                         nb_epoch=25,
                         validation_data=test_set,
                         nb_val_samples=2000)
