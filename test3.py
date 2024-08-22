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
marker_size = 200  # Marker size in pixels

# Use four different markers to define corners
marker_ids = [42, 43, 44, 45]
projector_points = []

# Generate and project four markers at random positions
for marker_id in marker_ids:
    marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
    aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img)
    
    # Save the marker image temporarily
    temp_dir = tempfile.gettempdir()
    marker_path = os.path.join(temp_dir, f'aruco_marker_{marker_id}.png')
    cv2.imwrite(marker_path, marker_img)

    # Load the marker image using Pygame
    marker_surface = pygame.image.load(marker_path)

    # Randomize the position on the screen
    random_x = random.randint(0, screen_width - marker_size)
    random_y = random.randint(0, screen_height - marker_size)
    projector_points.append([random_x, random_y])
    
    # Clear the screen and project the marker at the random location
    screen.fill((0, 0, 0))
    screen.blit(marker_surface, (random_x, random_y))
    pygame.display.flip()
    pygame.time.wait(1000)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize ArUco Detector
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Detect the markers with the camera
webcam_points = []
for _ in range(10):  # Retry to make sure all markers are detected
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Detect markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    
    # If markers detected
    if ids is not None:
        for marker_id in marker_ids:
            if marker_id in ids:
                index = np.where(ids == marker_id)[0][0]
                marker_corners = corners[index][0]
                detected_position = np.mean(marker_corners, axis=0)
                webcam_points.append(detected_position)
                print(f"Detected marker {marker_id} position: {detected_position}")

    if len(webcam_points) == len(marker_ids):
        break

if len(webcam_points) == len(marker_ids):
    # Calculate the homography matrix
    projector_points = np.array(projector_points, dtype="float32")
    webcam_points = np.array(webcam_points, dtype="float32")
    homography_matrix, _ = cv2.findHomography(webcam_points, projector_points)
else:
    print("Not all markers were detected. Unable to calculate homography.")
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()
    exit()

# Function to translate coordinates using the homography matrix
def translate_coordinates(point, to_projector=True):
    point = np.array([point], dtype="float32")
    if to_projector:
        projected_point = cv2.perspectiveTransform(np.array([point]), homography_matrix)
        return projected_point[0][0]
    else:
        inverse_homography_matrix = np.linalg.inv(homography_matrix)
        webcam_point = cv2.perspectiveTransform(np.array([point]), inverse_homography_matrix)
        return webcam_point[0][0]

# Example test with an example image
example_image_path = 'example_image.png'  # Replace with the path to your example image
example_image = pygame.image.load(example_image_path)

# Desired position in webcam coordinates
webcam_position = np.array([200, 100])

# Convert the webcam position to projector coordinates
projector_position = translate_coordinates(webcam_position, to_projector=True)
print(f"Webcam coordinates: {webcam_position}, Projector coordinates: {projector_position}")

# Project the example image at the calculated projector coordinates
screen.fill((0, 0, 0))
screen.blit(example_image, (int(projector_position[0]), int(projector_position[1])))
pygame.display.flip()

# Wait for 5 seconds to see the projection
pygame.time.wait(5000)

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()

# Remove the temporary files
for marker_id in marker_ids:
    os.remove(os.path.join(temp_dir, f'aruco_marker_{marker_id}.png'))
