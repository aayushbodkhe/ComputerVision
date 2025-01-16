import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def dice_coefficient(binary_image1, binary_image2):
    intersection = np.sum(binary_image1 * binary_image2)
    return (2. * intersection) / (np.sum(binary_image1) + np.sum(binary_image2))

def simple_matching_coefficient(binary_image1, binary_image2):
    matches = np.sum(binary_image1 == binary_image2)
    total_elements = binary_image1.size
    return matches / total_elements

# Load the image
image = cv2.imread(r'C:\cv\Dataset1\train\images\00056_129.jpg', 0)  # Load in grayscale

# Apply simple thresholding
_, thresholded_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Apply adaptive thresholding
adaptive_thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply Otsu's thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Region growing (simple example using floodFill)
seed_point = (0, 0)  # Starting point for region growing
region_growing_image = image.copy()
cv2.floodFill(region_growing_image, None, seed_point, 255)

# Calculate Dice coefficients
dice_thresholded = dice_coefficient(image, thresholded_image)
dice_adaptive = dice_coefficient(image, adaptive_thresh)
dice_otsu = dice_coefficient(image, otsu_thresh)
dice_region_growing = dice_coefficient(image, region_growing_image)

# Calculate Dice coefficients between different techniques
dice_thresh_adaptive = dice_coefficient(thresholded_image, adaptive_thresh)
dice_thresh_otsu = dice_coefficient(thresholded_image, otsu_thresh)
dice_thresh_region_growing = dice_coefficient(thresholded_image, region_growing_image)
dice_adaptive_otsu = dice_coefficient(adaptive_thresh, otsu_thresh)
dice_adaptive_region_growing = dice_coefficient(adaptive_thresh, region_growing_image)
dice_otsu_region_growing = dice_coefficient(otsu_thresh, region_growing_image)

# Calculate SSIM
ssim_thresholded = ssim(image, thresholded_image)
ssim_adaptive = ssim(image, adaptive_thresh)
ssim_otsu = ssim(image, otsu_thresh)
ssim_region_growing = ssim(image, region_growing_image)

# Calculate SMC
smc_thresholded = simple_matching_coefficient(image, thresholded_image)
smc_adaptive = simple_matching_coefficient(image, adaptive_thresh)
smc_otsu = simple_matching_coefficient(image, otsu_thresh)
smc_region_growing = simple_matching_coefficient(image, region_growing_image)

# Print Dice Coefficients
print(f'Dice Coefficient (Simple Thresholding): {dice_thresholded}')
print(f'Dice Coefficient (Adaptive Thresholding): {dice_adaptive}')
print(f'Dice Coefficient (Otsu Thresholding): {dice_otsu}')
print(f'Dice Coefficient (Region Growing): {dice_region_growing}')
print(f'Dice Coefficient (Thresholded vs Adaptive): {dice_thresh_adaptive}')
print(f'Dice Coefficient (Thresholded vs Otsu): {dice_thresh_otsu}')
print(f'Dice Coefficient (Thresholded vs Region Growing): {dice_thresh_region_growing}')
print(f'Dice Coefficient (Adaptive vs Otsu): {dice_adaptive_otsu}')
print(f'Dice Coefficient (Adaptive vs Region Growing): {dice_adaptive_region_growing}')
print(f'Dice Coefficient (Otsu vs Region Growing): {dice_otsu_region_growing}')

# Print SSIM
print(f'SSIM (Simple Thresholding): {ssim_thresholded}')
print(f'SSIM (Adaptive Thresholding): {ssim_adaptive}')
print(f'SSIM (Otsu Thresholding): {ssim_otsu}')
print(f'SSIM (Region Growing): {ssim_region_growing}')

# Print SMC
print(f'SMC (Simple Thresholding): {smc_thresholded}')
print(f'SMC (Adaptive Thresholding): {smc_adaptive}')
print(f'SMC (Otsu Thresholding): {smc_otsu}')
print(f'SMC (Region Growing): {smc_region_growing}')

# Save the thresholded images
cv2.imwrite('thresholded_image.jpg', thresholded_image)
cv2.imwrite('adaptive_thresh.jpg', adaptive_thresh)
cv2.imwrite('otsu_thresh.jpg', otsu_thresh)
cv2.imwrite('region_growing_image.jpg', region_growing_image)

# Display the original and thresholded images
cv2.imshow('Original Image', image)
cv2.imshow('Thresholded Image', thresholded_image)
cv2.imshow('Adaptive Thresholding', adaptive_thresh)
cv2.imshow('Otsu Thresholding', otsu_thresh)
cv2.imshow('Region Growing', region_growing_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
