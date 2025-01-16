import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
import os
import glob
import tensorflow as tf

# Load the images from your dataset directory
def load_images_from_folder(folder, img_size=(128, 128)):
    images = []
    for filename in glob.glob(os.path.join(folder, "*.png")):  # Change the extension if needed
        img = tf.keras.preprocessing.image.load_img(filename, target_size=img_size, color_mode="grayscale")
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)
    return np.array(images)

# Paths to your dataset
image_dir = r"C:\cv\data_red\images"  # Change to your actual image directory

# Load the dataset
x_train = load_images_from_folder(image_dir)
x_test = x_train  # For simplicity, using the same dataset for testing (you can split if needed)

# Add noise to the images
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

# Clip the values to be between 0 and 1
x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# Reshape the data for the model
x_train_noisy = np.reshape(x_train_noisy, (len(x_train_noisy), 128, 128, 1))  # Adjust based on your image size
x_test_noisy = np.reshape(x_test_noisy, (len(x_test_noisy), 128, 128, 1))
x_train = np.reshape(x_train, (len(x_train), 128, 128, 1))
x_test = np.reshape(x_test, (len(x_test), 128, 128, 1))

# Build the autoencoder
def build_autoencoder():
    model = keras.Sequential()
    
    # Encoder
    model.add(layers.Input(shape=(128, 128, 1)))  # Change to match your image size
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    
    # Decoder
    model.add(layers.Conv2D(8, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    
    model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
    
    return model

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(x_train_noisy, x_train,
                epochs=50,
                batch_size=128,
                validation_data=(x_test_noisy, x_test))

# Denoise the test images
denoised_images = autoencoder.predict(x_test_noisy)

# Plot original, noisy, and denoised images
n = 10  # Number of images to display
plt.figure(figsize=(20, 6))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i].reshape(128, 128), cmap='gray')  # Adjust based on your image size
    plt.title("Original")
    plt.axis("off")

    # Noisy images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i].reshape(128, 128), cmap='gray')
    plt.title("Noisy")
    plt.axis("off")

    # Denoised images
    ax = plt.subplot(3, n, i + 1 + 2*n)
    plt.imshow(denoised_images[i].reshape(128, 128), cmap='gray')
    plt.title("Denoised")
    plt.axis("off")

plt.show()
