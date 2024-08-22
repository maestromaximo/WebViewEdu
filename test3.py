import cv2
import numpy as np
import pygame
import cv2.aruco as aruco
import os
import shutil
import time

# Ensure debug_photos directory is clean
debug_photos = 'debug_photos'
if os.path.exists(debug_photos):
    shutil.rmtree(debug_photos)
os.makedirs(debug_photos)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create ArUco dictionary and generate the marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 42
marker_size = 200  # Marker size in pixels
marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img)

# Save the marker image for display
marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

# Initialize ArUco Detector
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Detect the marker with the camera
detected_position = None
difference = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break
    
    # Detect markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    
    # If marker detected
    if ids is not None and marker_id in ids:
        index = np.where(ids == marker_id)[0][0]
        marker_corners = corners[index][0]
        
        # Assume the first corner (top-left) as the detected position
        detected_position = np.mean(marker_corners, axis=0)
        print(f"Detected marker position: {detected_position}")
        
        # Assume marker was projected at the center of the screen
        screen_center = np.array([marker_size / 2, marker_size / 2])
        
        # Calculate the difference
        difference = detected_position - screen_center
        print(f"Difference between projected and detected: {difference}")
        
        break
    else:
        print("No marker detected, continuing...")

# Function to translate coordinates between camera and projector
def translate_coordinates(point, to_projector=True):
    if difference is None:
        print("No difference calculated, cannot translate coordinates.")
        return point
    if to_projector:
        return point - difference
    else:
        return point + difference

# Initialize Pygame for projection
pygame.init()
screen = pygame.display.set_mode((1920, 1080))

# Display the marker in Pygame
marker_surface = pygame.surfarray.make_surface(marker_img_bgr.swapaxes(0, 1))
screen.blit(marker_surface, (1920//2 - marker_size//2, 1080//2 - marker_size//2))
pygame.display.flip()

time.sleep(4)  # Allow time for the marker to be stable on screen

# Project an image using the calculated translation
if detected_position is not None:
    print("Using detected position for projection")

    # Define the square in the right middle of the camera view
    square_size = 300  # Size of the square
    screen_width, screen_height = 1920, 1080

    # Define target square in the right middle of the screen
    target_square_corners = np.array([
        [screen_width - square_size - 50, screen_height // 2 - square_size // 2],  # Top-left
        [screen_width - 50, screen_height // 2 - square_size // 2],  # Top-right
        [screen_width - 50, screen_height // 2 + square_size // 2],  # Bottom-right
        [screen_width - square_size - 50, screen_height // 2 + square_size // 2]  # Bottom-left
    ])

    # Translate corners using the detected difference
    translated_corners = np.array([translate_coordinates(corner) for corner in target_square_corners])
    print(f"Translated corners for projection: {translated_corners}")

    # Load an image for projection
    img = cv2.imread('example_image.png')
    if img is None:
        print("Failed to load image.")
    else:
        img_height, img_width = img.shape[:2]
        img_corners = np.float32([[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]])

        # Compute the transformation matrix
        matrix = cv2.getPerspectiveTransform(img_corners, translated_corners)
        transformed_img = cv2.warpPerspective(img, matrix, (1920, 1080))

        # Display the transformed image using Pygame
        transformed_surface = pygame.surfarray.make_surface(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
        screen.blit(transformed_surface, (0, 0))
        pygame.display.flip()
        print("Image projected within the defined square.")
else:
    print("Failed to detect ArUco marker.")

# Keep the window open until closed by the user
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()
