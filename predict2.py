import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet152
from keras.applications.resnet50 import preprocess_input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2M,preprocess_input
import numpy as np
import cv2

# Load the trained model
model_path = '/home/dip_21/weight/effnetv2.h5'
model = load_model(model_path)

# Load and preprocess a single image for prediction
image_path = '/home/dip_21/project/CCSN_v2/Ci/Ci-N001.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.resize(image, (224, 224))
image = preprocess_input(image)
image = np.expand_dims(image, axis=0)  # Add batch dimension

# Perform prediction
preds = model.predict(image)
predicted_class = np.argmax(preds, axis=1)

# Get the class labels
class_indices = {0: 'class_0', 1: 'class_1', 2: 'class_2', 3: 'class_3', 4: 'class_4', 5: 'class_5', 6: 'class_6'}

# Get the predicted class label
predicted_label = class_indices[predicted_class[0]]

print(f'Predicted class: {predicted_class}')