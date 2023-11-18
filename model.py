# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten

# Set the path to your dataset
dataset_path = r'C:/Users/khana/OneDrive/Desktop/Fast/bone/bone-cancer-classifier/Datasets'

# Define parameters
input_shape = (150, 150, 3)  # adjust the size based on your images
batch_size = 32
epochs = 10

# Data preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(dataset_path, 'train'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',  # Since it's a binary classification task
    classes=['normal', 'malignant']  # Specify class names
)

val_generator = val_datagen.flow_from_directory(
    os.path.join(dataset_path, 'val'),
    target_size=input_shape[:2],
    batch_size=batch_size,
    class_mode='binary',
    classes=['normal', 'malignant']
)

# Load pre-trained VGG16 model without the top classification layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Build the model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=len(val_generator)
)

# Save the model
model.save('model.h5')

# Evaluate model on validation data
val_loss, val_accuracy = model.evaluate(val_generator, steps=len(val_generator))
print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

# Make predictions on a sample test image
sample_image_path = 'C:/Users/khana/OneDrive/Desktop/Fast/bone/bone-cancer-classifier/Datasets/val/malignant/m (2).jpg'
sample_image = tf.keras.preprocessing.image.load_img(sample_image_path, target_size=(150, 150))
sample_image_array = tf.keras.preprocessing.image.img_to_array(sample_image) / 255.0
sample_image_array = np.expand_dims(sample_image_array, axis=0)

prediction = model.predict(sample_image_array)
predicted_class = 'malignant' if prediction[0, 0] > 0.5 else 'normal'

print(f'Prediction: {predicted_class}, Confidence: {prediction[0, 0]:.4f}')
