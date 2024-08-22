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

# Create and save the marker only once
marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img)
temp_dir = tempfile.gettempdir()
marker_path = os.path.join(temp_dir, f'aruco_marker_{marker_id}.png')
cv2.imwrite(marker_path, marker_img)
marker_surface = pygame.image.load(marker_path)

# Initialize the camera
cap = cv2.VideoCapture(0)

# Optional: Reduce frame size to speed up processing (if appropriate for your setup)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize ArUco Detector
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Function to check if four points form a rectangle
def is_rectangle(points):
    if len(points) != 4:
        return False

    def distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    d1 = distance(points[0], points[1])
    d2 = distance(points[1], points[2])
    d3 = distance(points[2], points[3])
    d4 = distance(points[3], points[0])

    diag1 = distance(points[0], points[2])
    diag2 = distance(points[1], points[3])

    return np.isclose(d1, d3) and np.isclose(d2, d4) and np.isclose(diag1, diag2)

# Function to project and detect the marker
def project_and_detect(x, y, attempts=3, timeout=5):
    detected_position = None
    
    # Project the marker at the given coordinates
    screen.fill((0, 0, 0))
    screen.blit(marker_surface, (x, y))
    pygame.display.flip()

    # Try to detect the marker within the given timeout
    for _ in range(attempts):
        start_time = time.time()
        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect markers
            corners, ids, rejected = aruco_detector.detectMarkers(frame)
            
            if ids is not None and marker_id in ids:
                index = np.where(ids == marker_id)[0][0]
                marker_corners = corners[index][0]
                detected_position = np.mean(marker_corners, axis=0)
                print(f"Detected marker at position: {detected_position}")
                return detected_position

    print("Marker not detected after several attempts.")
    return detected_position

# Example set of four webcam coordinates (example rectangle)
webcam_rect_coords = [(100, 100), (400, 100), (400, 300), (100, 300)]

# Check if the provided coordinates form a rectangle
if not is_rectangle(webcam_rect_coords):
    print("Provided coordinates do not form a valid rectangle.")
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()
    exit()

# Define the center of the projection
center_x = (screen_width - marker_size) // 2
center_y = (screen_height - marker_size) // 2

# Project at the center first
print("Projecting at the center of the projection screen.")
center_position = project_and_detect(center_x, center_y)

if center_position is None:
    print("Failed to detect marker at the center.")
    cap.release()
    pygame.quit()
    cv2.destroyAllWindows()
    exit()

# Move marker to each corner of the rectangle and save the corresponding projector coordinates
projector_rect_coords = []

for i, (wx, wy) in enumerate(webcam_rect_coords):
    print(f"Moving marker to match webcam rectangle corner {i + 1}")
    
    # Calculate the offset to move the marker towards the webcam corner
    offset_x = wx - center_position[0]
    offset_y = wy - center_position[1]
    
    # Project the marker at the calculated offset
    moved_x = int(center_x + offset_x)
    moved_y = int(center_y + offset_y)
    
    final_position = project_and_detect(moved_x, moved_y)
    
    if final_position is not None:
        projector_rect_coords.append((moved_x, moved_y))
    else:
        print(f"Failed to match corner {i + 1} after moving marker.")

# Print the projector coordinates for the rectangle
print("Projector coordinates for the rectangle corners:")
for i, coord in enumerate(projector_rect_coords):
    print(f"Corner {i + 1}: {coord}")

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()

# Remove the temporary file
os.remove(marker_path)
