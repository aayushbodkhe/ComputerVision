import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import os
import glob

# Preprocess the input image
def preprocess_image(image_path, img_size=(128, 128)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=img_size)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Predict mask for fire
def predict_mask(model, image_path, img_size=(128, 128)):
    img = preprocess_image(image_path, img_size)
    pred_mask = model.predict(img)
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold the mask
    return pred_mask[0]

# Display input image, predicted mask, and actual mask (if provided)
def display_result(image_path, pred_mask, actual_mask_path=None):
    img = tf.keras.preprocessing.image.load_img(image_path)
    img = tf.keras.preprocessing.image.img_to_array(img) / 255.0

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Input Image')
    plt.imshow(img)

    plt.subplot(1, 3, 2)
    plt.title('Predicted Mask')
    plt.imshow(pred_mask.squeeze(), cmap='gray')

    if actual_mask_path:
        actual_mask = tf.keras.preprocessing.image.load_img(actual_mask_path, color_mode='grayscale')
        actual_mask = tf.keras.preprocessing.image.img_to_array(actual_mask) / 255.0
        plt.subplot(1, 3, 3)
        plt.title('Actual Mask')
        plt.imshow(actual_mask.squeeze(), cmap='gray')

    plt.show()

# UNet model definition for fire segmentation
def unet_model(input_size=(128, 128, 3)):
    inputs = layers.Input(input_size)

    # Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Example usage for training
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Load and preprocess the dataset from directories
image_dir = r"D:\code\CV_DL\assg_9\Image\Fire" # Folder path for images
mask_dir = r"D:\code\CV_DL\assg_9\Segmentation_Mask\Fire"   # Folder path for masks

# Load data function
def load_data_from_folder(image_dir, mask_dir, img_size=(128, 128)):
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))  # Change the extension if needed
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))    # Change the extension if needed

    images = []
    masks = []
    
    for img_path, mask_path in zip(image_paths, mask_paths):
        assert os.path.exists(img_path), f"Image not found: {img_path}"
        assert os.path.exists(mask_path), f"Mask not found: {mask_path}"
        
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=img_size)
        img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        mask = tf.keras.preprocessing.image.load_img(mask_path, target_size=img_size, color_mode="grayscale")
        mask = tf.keras.preprocessing.image.img_to_array(mask) / 255.0
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

# Load the dataset
X, y = load_data_from_folder(image_dir, mask_dir)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=16)

# Save the trained model
model.save('unet_fire_segmentation.h5')

# Example usage for prediction and display
new_image_path = r"D:\code\CV_DL\assg_9\Image\Fire\Img_1315.jpg"
actual_mask_path = r"D:\code\CV_DL\assg_9\Segmentation_Mask\Fire\Img_1315.jpg"
predicted_mask = predict_mask(model, new_image_path)
display_result(new_image_path, predicted_mask, actual_mask_path)

