import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D,Conv2DTranspose,GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet152
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import wandb


print('clean data')
wandb.init(
    project="ThaiSlik"

)


datagen = ImageDataGenerator(
                                rescale=1 / 255.,
                                rotation_range=45,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=0.2,
                                horizontal_flip=True,
                                vertical_flip=True,
                                fill_mode='nearest',
                                preprocessing_function=preprocess_input
                             )

train_generator = datagen.flow_from_directory(
        '/home/dip_21/project/split_statified/train',
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical")
validation_generator = datagen.flow_from_directory(
        '/home/dip_21/project/split_statified/val',
        target_size=(224, 224),
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical")



train_dir = '/home/dip_21/project/split_statified/train'
classes = os.listdir(train_dir)
class_counts = {}

# Mapping class names to indices
class_to_index = {class_name: idx for idx, class_name in enumerate(classes)}

for class_name in classes:
    class_path = os.path.join(train_dir, class_name)
    num_samples = len(os.listdir(class_path))
    class_counts[class_name] = num_samples

total_samples = sum(class_counts.values())

# Creating class weights with indices as keys
class_weights = {class_to_index[class_name]: total_samples / (num_samples * len(class_counts)) for class_name, num_samples in class_counts.items()}

print(f'class weight : {class_weights}')
# Create a Sequential model
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
model.add(Dense(4, activation='softmax'))

# Compile the model
opt = RMSprop(learning_rate=0.0005)
model.compile(optimizer=opt,
            loss='categorical_crossentropy',
            metrics=['accuracy'])
model.summary()

wandb_callback = wandb.keras.WandbCallback(log_weights=True)
step_size_train=train_generator.n//train_generator.batch_size
step_size_val=validation_generator.n//validation_generator.batch_size
checkpoint = ModelCheckpoint("/home/dip_21/project/checkpoint/resnet_th", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False)
early = EarlyStopping(monitor='val_acc', patience=8, verbose=1)
start_time = time.time()
resnet_history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = step_size_train,
                            validation_steps = step_size_val,
                            class_weight=class_weights,
                            epochs = 120, callbacks=[checkpoint,wandb_callback])
end_time = time.time()
elapsed_time = end_time - start_time
print(f'time : {elapsed_time}')
# Performance Visualization
N = range(1, len(resnet_history.history["accuracy"])+1)
# View Accuracy (Training, Validation)
plt.figure(figsize=(10, 6))
plt.plot(N,resnet_history.history["accuracy"], label="Train_acc")
plt.plot(N,resnet_history.history["val_accuracy"], label="Validate_acc")
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f'/home/dip_21/project/plot/acc_resnet_th.png')

# View Loss (Training, Validation)
plt.figure(figsize=(10, 6))
plt.plot(N,resnet_history.history['loss'], label="Train_loss")
plt.plot(N,resnet_history.history['val_loss'], label="Validate_loss")
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f'/home/dip_21/project/plot/loss_resnet_th.png')


