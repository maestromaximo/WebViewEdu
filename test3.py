import cv2
import numpy as np
import cv2.aruco as aruco
import pygame
import tempfile
import os
import time

# Initialize pygame for projection
pygame.init()
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# Create ArUco dictionary and generate the marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_size = 200  # Marker size in pixels
marker_id = 42  # Reusing the same marker ID

# Set the number of points to use for homography
num_points = 12  # Adjust the number of points as needed

projector_points = []
webcam_points = []

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize ArUco Detector
detector_params = aruco.DetectorParameters()
detector_params.minMarkerPerimeterRate = 0.03  # Adjusted perimeter rate for better detection
detector_params.maxMarkerPerimeterRate = 4.0
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Process each marker at different positions
for _ in range(num_points):  # Adjust number of positions to the variable `num_points`
    detected = False
    attempt_count = 0
    max_attempts = 5  # Limit the number of regeneration attempts
    
    while not detected and attempt_count < max_attempts:
        attempt_count += 1
        
        marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
        aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img)
        
        # Save the marker image temporarily
        temp_dir = tempfile.gettempdir()
        marker_path = os.path.join(temp_dir, f'aruco_marker_{marker_id}.png')
        cv2.imwrite(marker_path, marker_img)

        # Load the marker image using Pygame
        marker_surface = pygame.image.load(marker_path)

        # Randomize the position on the screen
        random_x = np.random.randint(0, screen_width - marker_size)
        random_y = np.random.randint(0, screen_height - marker_size)
        projector_points.append([random_x, random_y])
        
        # Display the marker on the screen
        screen.fill((0, 0, 0))
        screen.blit(marker_surface, (random_x, random_y))
        pygame.display.flip()

        # Start timer
        start_time = time.time()

        # Try to detect the marker
        while time.time() - start_time < 7:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect markers
            corners, ids, rejected = aruco_detector.detectMarkers(frame)
            
            # If the specific marker is detected
            if ids is not None and marker_id in ids:
                index = np.where(ids == marker_id)[0][0]
                marker_corners = corners[index][0]
                detected_position = np.mean(marker_corners, axis=0)
                webcam_points.append(detected_position)
                print(f"Detected marker at position: {detected_position}")
                detected = True
                break

        # If the marker wasn't detected in 7 seconds, regenerate its position
        if not detected:
            print(f"Marker not detected in 7 seconds, regenerating position.")
            projector_points.pop()  # Remove the last added point, as it's not valid

    if not detected:
        print(f"Marker failed after {max_attempts} attempts. Moving on.")

# Calculate the homography matrix
if len(webcam_points) == num_points:  # Expecting the specified number of points for homography
    projector_points = np.array(projector_points, dtype="float32")
    webcam_points = np.array(webcam_points, dtype="float32")
    homography_matrix, _ = cv2.findHomography(webcam_points, projector_points)
else:
    print("Not all points were detected. Unable to calculate homography.")
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
os.remove(marker_path)
