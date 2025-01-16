import cv2
import numpy as np

# Load the image
img = cv2.imread(r"D:\Pccoe doc\Projects\AIDER\flooded_areas\flood_image0344.jpg")


def rotate(img):
    rows, cols = img.shape[:2]
    img_rotation = cv2.warpAffine(img, cv2.getRotationMatrix2D((cols/2, rows/2), 30, 0.6), (cols, rows))
    cv2.imshow('Rotated Image', img_rotation)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def translate(img):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('Translated Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def affine_transformation(img):
    rows, cols = img.shape[:2]
    pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow('Affine Transformed Image', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def crop(img):
    cropped_img = img[100:300, 100:300]
    cv2.imshow('Cropped Image', cropped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def shear(img):
    rows, cols = img.shape[:2]
    M = np.float32([[1, 0, 0], [0.5, 1, 0], [0, 0, 1]])
    sheared_img = cv2.warpPerspective(img, M, (int(cols * 1.5), int(rows * 1.5)))
    cv2.imshow('Sheared Image', sheared_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def menu():
    while True:
        print("\nMenu:")
        print("1. Rotate Image")
        print("2. Translate Image")
        print("3. Affine Transformation")
        print("4. Shear Image")
        print("5. Crop Image")
        print("6. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            rotate(img)
        elif choice == '2':
            translate(img)
        elif choice == '3':
            affine_transformation(img)
        elif choice == '4':
            shear(img)
        elif choice == '5':
            crop(img)
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    menu()
