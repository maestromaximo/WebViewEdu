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
marker_size = 300  # Marker size in pixels

# Set the number of points to use for homography
num_points = 10

projector_points = []
webcam_points = []

# Initialize the camera
cap = cv2.VideoCapture(0)

# Initialize ArUco Detector with tuned parameters
detector_params = aruco.DetectorParameters()
detector_params.minMarkerPerimeterRate = 0.03
detector_params.maxMarkerPerimeterRate = 4.0
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

temp_dir = tempfile.gettempdir()

def display_and_detect_markers():
    global projector_points

    for _ in range(num_points):
        # Remove any previous marker files
        for i in range(4):
            marker_path = os.path.join(temp_dir, f'aruco_marker_{i}.png')
            if os.path.exists(marker_path):
                os.remove(marker_path)
        
        # Display 4 markers
        marker_positions = []
        for i in range(4):
            random_x = np.random.randint(0, screen_width - marker_size)
            random_y = np.random.randint(0, screen_height - marker_size)
            marker_positions.append((random_x, random_y))
            projector_points.append([random_x, random_y])

            # Generate and save the marker image temporarily
            marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
            aruco.generateImageMarker(aruco_dict, i, marker_size, marker_img)
            
            marker_path = os.path.join(temp_dir, f'aruco_marker_{i}.png')
            cv2.imwrite(marker_path, marker_img)

            # Load the marker image using Pygame
            marker_surface = pygame.image.load(marker_path)
            screen.blit(marker_surface, (random_x, random_y))

        pygame.display.flip()

        # Attempt to detect the markers
        start_time = time.time()
        detected_points = 0
        while time.time() - start_time < 7:
            ret, frame = cap.read()
            if not ret:
                continue
            
            corners, ids, rejected = aruco_detector.detectMarkers(frame)
            if ids is not None:
                detected_points = len(ids)
                for id in ids:
                    index = np.where(ids == id)[0][0]
                    marker_corners = corners[index][0]
                    detected_position = np.mean(marker_corners, axis=0)
                    webcam_points.append(detected_position)
                break

        if detected_points >= 3:
            break  # Move on if 3 or more markers are detected
        elif detected_points > 0:
            print(f"Only {detected_points} markers detected, adding their positions.")
        else:
            print("No markers detected, regenerating positions.")
            projector_points = projector_points[:-4]  # Remove the last 4 points

    return len(webcam_points)

# Run the detection loop and calculate the homography
total_detected_points = display_and_detect_markers()

if total_detected_points >= 4:
    projector_points = np.array(projector_points[:total_detected_points], dtype="float32")
    webcam_points = np.array(webcam_points, dtype="float32")
    homography_matrix, _ = cv2.findHomography(webcam_points, projector_points, cv2.RANSAC)
    print("Homography matrix calculated successfully.")
else:
    print("Not enough points were detected. Unable to calculate homography.")
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

# Function to detect yellow objects
def detect_yellow_center(frame):
    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a binary mask where yellow colors are white and the rest are black
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the main yellow object
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the moments to calculate the centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    
    return None  # Return None if no yellow object is detected

# Main loop for yellow detection and projection
while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect yellow center in the frame
    yellow_center = detect_yellow_center(frame)

    if yellow_center is not None:
        print(f"Yellow object detected at: {yellow_center}")
        
        # Convert the yellow center from webcam to projector coordinates
        projector_position = translate_coordinates(yellow_center, to_projector=True)
        print(f"Projector coordinates: {projector_position}")

        # Project a green circle at the calculated projector coordinates
        screen.fill((0, 0, 0))
        pygame.draw.circle(screen, (0, 255, 0), (int(projector_position[0]), int(projector_position[1])), 10)
        pygame.display.flip()
    else:
        print("No yellow object detected")

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
pygame.quit()
cv2.destroyAllWindows()

# Remove the temporary files
for i in range(4):
    marker_path = os.path.join(temp_dir, f'aruco_marker_{i}.png')
    if os.path.exists(marker_path):
        os.remove(marker_path)
