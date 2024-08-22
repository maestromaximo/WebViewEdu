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

# Detect the marker with the camera multiple times for stability
num_detections = 10
detected_positions = []

for _ in range(num_detections):
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Detect markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    
    # If marker detected
    if ids is not None and marker_id in ids:
        index = np.where(ids == marker_id)[0][0]
        marker_corners = corners[index][0]
        
        # Assume the first corner (top-left) as the detected position
        detected_position = np.mean(marker_corners, axis=0)
        detected_positions.append(detected_position)

# Calculate the average detected position
if detected_positions:
    avg_detected_position = np.mean(detected_positions, axis=0)
    print(f"Detected marker position: {avg_detected_position}")
else:
    avg_detected_position = None
    print("Marker not detected.")

# Print the projected and detected positions
print(f"Projected position: ({random_x}, {random_y})")
if avg_detected_position is not None:
    difference = avg_detected_position - np.array([random_x, random_y])
    print(f"Difference between projected and detected: {difference}")
else:
    print("Marker not detected.")
    difference = np.array([0, 0])  # Fallback if not detected

# Function to translate coordinates between camera and projector
def translate_coordinates(point, to_projector=True):
    if to_projector:
        return point - difference
    else:
        return point + difference

# Example test of the function with an example image
example_image_path = 'example_image.png'  # Replace with the path to your example image
example_image = pygame.image.load(example_image_path)

# Desired position in webcam coordinates
webcam_position = np.array([200, 100])

# Convert the webcam position to projector coordinates
projector_position = translate_coordinates(webcam_position, to_projector=True)
print(f"Webcam coordinates: {webcam_position}, Projector coordinates: {projector_position}")

# Project the example image at the calculated projector coordinates
screen.fill((0, 0, 0))
screen.blit(example_image, projector_position)
pygame.display.flip()

# Wait for 5 seconds to see the projection
pygame.time.wait(5000)

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()

# Remove the temporary file
os.remove(marker_path)
