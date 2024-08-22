import cv2
import numpy as np
import pygame
import os
import shutil
import time

# Ensure debug_photos directory is clean
debug_photos = 'debug_photos'
if os.path.exists(debug_photos):
    shutil.rmtree(debug_photos)
os.makedirs(debug_photos)

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()

# Function to detect ArUco marker
def detect_aruco_marker(debug=False):
    cap = cv2.VideoCapture(0)  # Adjust the device index if needed
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture video frame")
        if debug:
            cv2.imwrite('debug_frame.jpg', frame)
        return None

    corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict, parameters=aruco_params)
    if ids is not None:
        if debug:
            cv2.imwrite('debug_frame.jpg', frame)
        return corners[0][0].tolist()  # Return the first detected marker's corners
    else:
        if debug:
            cv2.imwrite('debug_frame.jpg', frame)
        print("ArUco marker not detected. Check debug_frame.jpg for analysis.")
        return None

# Function to create the mapping function
def create_mapping_to_square(detected_corners, target_square_corners):
    matrix = cv2.getPerspectiveTransform(np.array(detected_corners, dtype=np.float32), np.array(target_square_corners, dtype=np.float32))
    return matrix

# Function to project an image using a mapping function
def project_image(screen, matrix, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return

    transformed_img = cv2.warpPerspective(img, matrix, (1920, 1080))
    transformed_surface = pygame.surfarray.make_surface(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    screen.blit(transformed_surface, (0, 0))
    pygame.display.flip()
    print("Image projected within the defined square.")

def main():
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    board_pos = (300, 200)  # Position where ArUco marker is projected

    # Generate an ArUco marker and project it
    marker_id = 0
    marker_size = 600  # Size of the marker
    marker_img = cv2.aruco.drawMarker(aruco_dict, marker_id, marker_size)
    marker_surface = pygame.surfarray.make_surface(cv2.cvtColor(marker_img, cv2.COLOR_GRAY2RGB).swapaxes(0, 1))
    screen.blit(marker_surface, board_pos)
    pygame.display.flip()

    time.sleep(4)  # Allow time for the marker to be stable on screen

    detected_corners = detect_aruco_marker(debug=True)
    if detected_corners:
        print(f"Detected ArUco marker at: {detected_corners}")

        # Define the square in the right middle of the camera view
        square_size = 300  # Size of the square
        screen_width, screen_height = 1920, 1080

        # Define target square in the right middle of the screen
        target_square_corners = [
            [screen_width - square_size - 50, screen_height // 2 - square_size // 2],  # Top-left
            [screen_width - 50, screen_height // 2 - square_size // 2],  # Top-right
            [screen_width - 50, screen_height // 2 + square_size // 2],  # Bottom-right
            [screen_width - square_size - 50, screen_height // 2 + square_size // 2]  # Bottom-left
        ]

        transform_matrix = create_mapping_to_square(detected_corners, target_square_corners)
        project_image(screen, transform_matrix, 'example_image.png')
    else:
        print("Failed to detect ArUco marker.")

    # Keep the window open until closed by the user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()
