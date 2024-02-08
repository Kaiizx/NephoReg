#srun --gres=gpu -c 8 --mem 16G --time 1-0 --pty /bin/bash

import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D,Conv2DTranspose,GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet152
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import splitfolders

datagen=ImageDataGenerator( 
    preprocessing_function=preprocess_input,
    fill_mode="nearest")

test_generator = datagen.flow_from_directory('/home/dip_21/project/CCSN_c/test',
    class_mode="categorical",
    target_size=(224, 224), color_mode="rgb",
    shuffle=False,
    batch_size=1)

model = Sequential()

# Add the ResNet152 base model (excluding the top layers)
base_model = ResNet152(weights='imagenet',include_top=False, input_shape=(224, 224, 3))
# base_model.trainable = False

model.add(base_model)
# for layer in model.layers[0].layers[:413]:
#                     layer.trainable = False

# Add custom dense layers
model.add(GlobalAveragePooling2D())
model.add(Dense(1024, activation='relu'))
# model.add(Dense(512, activation='relu'))
# model.add(Dense(256, activation='relu'))

# Add the final dense layer
# model.add(Dense(12, activation='softmax'))
model.add(Dense(7, activation='softmax'))
opt = RMSprop(learning_rate=0.0005)
model.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])

filename = '/home/dip_21/project/checkpoint/resnet_c'

print(f'Model : {filename}')
print('=====================================================\n\n')
model = load_model(filename)
model.summary
# model.load_weights(filename)
y_true = test_generator.classes
print(f"classes : {test_generator.class_indices}")

preds = model.predict(test_generator)
print(f'preds : {preds}')
y_pred = np.argmax(preds,axis=1)
print(f'class : {test_generator.classes}')
print(f'y_pred : {y_pred}')
print()
print('===================confusion_matrix==================\n')
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print('\n\n=====================================================\n\n')