import cv2
import numpy as np
import cv2.aruco as aruco
import pygame
import tempfile
import random
import os

# Initialize pygame for projection
pygame.init()
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# Create ArUco dictionary and generate the marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 42
marker_size = 200  # Marker size in pixels
marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img)

# Save the marker image temporarily
temp_dir = tempfile.gettempdir()
marker_path = os.path.join(temp_dir, 'aruco_marker.png')
cv2.imwrite(marker_path, marker_img)

# Load the marker image using Pygame
marker_surface = pygame.image.load(marker_path)

# Randomize the position on the screen
random_x = random.randint(0, screen_width - marker_size)
random_y = random.randint(0, screen_height - marker_size)

# Clear the screen and project the marker at the random location
screen.fill((0, 0, 0))
screen.blit(marker_surface, (random_x, random_y))
pygame.display.flip()

# Wait for 2 seconds to ensure the projection is visible
pygame.time.wait(2000)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize ArUco Detector
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Detect the marker with the camera
detected_position = None
while True:
    ret, frame = cap.read()
    if not ret:
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
        
        break

# Print the projected and detected positions
print(f"Projected position: ({random_x}, {random_y})")
if detected_position is not None:
    print(f"Difference between projected and detected: {detected_position - np.array([random_x, random_y])}")
else:
    print("Marker not detected.")

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()

# Remove the temporary file
os.remove(marker_path)
