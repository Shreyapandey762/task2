import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# Corrected path to the folder containing images directly
dataset_path = r'C:\Users\shrey\Downloads\val2017\val2017'

# Load images
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):  # Check if it's a file
            try:
                img = load_img(file_path, target_size=(128, 128))
                img_array = img_to_array(img)
                images.append(img_array)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return np.array(images)

# Load images
images = load_images_from_folder(dataset_path)
print(f"Number of images loaded: {len(images)}")

# Visualize a few images
def display_samples(images, num=5):
    plt.figure(figsize=(10, 10))
    for i in range(num):
        plt.subplot(1, num, i+1)
        plt.imshow(images[i].astype('uint8'))
        plt.axis('off')
    plt.show()

display_samples(images)

# Define U-Net model
def unet_model(input_size=(128, 128, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    u = UpSampling2D(size=(2, 2))(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(u)
    
    u2 = UpSampling2D(size=(2, 2))(c3)  # Add another upsampling layer
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u2)
    
    outputs = Conv2D(1, 1, activation='sigmoid')(c4)
    model = Model(inputs=[inputs], outputs=[outputs])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Initialize and compile the model
model = unet_model()
model.summary()

# Simulate masks for demonstration (replace this with actual mask data loading)
masks = np.random.randint(2, size=(5000, 128, 128, 1))

# Split the images into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, accuracy = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy}")
