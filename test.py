import cv2
import numpy as np
import time
import pygame
import os
import shutil
import threading

# Load environment variables
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Hyperparameters
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 2.5
DEBUG_VIEW = False
DEBUG_FEED = True
VIDEO_PATH = 'C:/Users/aleja/OneDrive/Escritorio/WebViewEdu/test_vid1.mp4'
AREA_THRESHOLD = 10
DISTANCE_THRESHOLD = 1
STABILITY_THRESHOLD = 30
CONFIRMATION_TIME = 10
REAPPEARANCE_TIME = 0.5
CUSHION = 10
WRITING_CHECK_INTERVAL = 5
TEXT_DETECTION_URL = "https://aleale2423-textdetector.hf.space/detect"
MARGIN = 10
AREA_CHANGE_THRESHOLD = 0.2
DETAIL_LEVEL = "low"
ORANGE_THRESHOLD = 50
DEBUG_BOARD = True  # Set this to True to simulate an average board
DEBUG_FOLDER_PHOTOS = True  # Set this to True to save processed images with detected circles
DEBUG_FOLDER_PATH = 'debug_photos'  # Path to the folder where debug photos will be saved

orange_count = 0

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def normalize_points(points):
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    normalized = []
    for x, y in points:
        nx = (x - min_x) / (max_x - min_x)
        ny = (y - min_y) / (max_y - min_y)
        normalized.append((nx, ny))

    return np.array(normalized, dtype=np.float32)

# Function to project circles at the corners
def project_circles(positions):
    height, width = 1080, 1920  # Assume 1080p resolution for the projector
    image = np.zeros((height, width, 3), np.uint8)
    color = (255, 0, 255)  # Purple color in BGR
    radius = 50  # Slightly larger radius for better visibility
    thickness = -1  # Filled circles

    for (x, y) in positions:
        cv2.circle(image, (x, y), radius, color, thickness)

    # Convert the image to a format suitable for pygame
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rot90(image)  # Rotate the image for correct orientation in pygame
    image = pygame.surfarray.make_surface(image)

    # Initialize pygame and display the image fullscreen
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen.blit(image, (0, 0))
    pygame.display.flip()

    return screen

# Function to detect circles and calculate the transformation
def detect_circles_and_calculate_transform(screen, positions):
    image_counter = 0

    # Clear the debug folder if it exists
    if DEBUG_FOLDER_PHOTOS:
        if os.path.exists(DEBUG_FOLDER_PATH):
            shutil.rmtree(DEBUG_FOLDER_PATH)
        os.makedirs(DEBUG_FOLDER_PATH)

    while True:
        # Capture image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture image from webcam")
            continue

        print("Captured image from webcam")

        # Save the original captured frame
        if DEBUG_FOLDER_PHOTOS:
            original_filename = os.path.join(DEBUG_FOLDER_PATH, f"original_frame_{image_counter}.png")
            cv2.imwrite(original_filename, frame)
            print(f"Saved original frame: {original_filename}")

        # Initialize the HSV range
        lower_h, upper_h = 120, 150
        lower_s, upper_s = 50, 255
        lower_v, upper_v = 50, 255
        step = 10

        while True:
            # Convert image to HSV and apply color mask for purple
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_purple = np.array([lower_h, lower_s, lower_v])
            upper_purple = np.array([upper_h, upper_s, upper_v])
            mask = cv2.inRange(hsv, lower_purple, upper_purple)

            # Apply the mask to get a binary image
            res = cv2.bitwise_and(frame, frame, mask=mask)
            gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

            print("Applied color mask and converted to binary image")

            # Save the binary frame
            if DEBUG_FOLDER_PHOTOS:
                binary_filename = os.path.join(DEBUG_FOLDER_PATH, f"binary_frame_{image_counter}.png")
                cv2.imwrite(binary_filename, binary)
                print(f"Saved binary frame: {binary_filename}")

            # Find contours in the binary image
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            detected_positions = []
            for contour in contours:
                if cv2.contourArea(contour) > AREA_THRESHOLD:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        detected_positions.append((cX, cY))

            if len(detected_positions) == 4:
                print(f"Detected 4 positions: {detected_positions}")
                # Draw green dots at the center of detected positions
                for (x, y) in detected_positions:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
                break

            print("No circles detected or insufficient number of circles, expanding HSV range")

            # Expand the HSV range
            lower_h = max(0, lower_h - step)
            upper_h = min(180, upper_h + step)
            lower_s = max(0, lower_s - step)
            upper_s = min(255, upper_s + step)
            lower_v = max(0, lower_v - step)
            upper_v = min(255, upper_v + step)

            # Save the processed frame with detected circles
            if DEBUG_FOLDER_PHOTOS:
                processed_filename = os.path.join(DEBUG_FOLDER_PATH, f"processed_frame_{image_counter}.png")
                cv2.imwrite(processed_filename, frame)
                print(f"Saved processed frame: {processed_filename}")
                image_counter += 1

            time.sleep(0.5)

        # Expected positions of the circles (adjusted inward)
        height, width = frame.shape[:2]
        offset = 100
        expected_positions = [(offset, offset), (width - offset, offset), (offset, height - offset), (width - offset, height - offset)]

        # Order points to ensure consistency
        detected_positions = order_points(np.array(detected_positions, dtype=np.float32))
        expected_positions = order_points(np.array(expected_positions, dtype=np.float32))

        # Normalize points
        src_pts = normalize_points(detected_positions)
        dst_pts = normalize_points(expected_positions)

        # Calculate transformation
        if len(detected_positions) == 4:
            transform_matrix, _ = cv2.findHomography(src_pts, dst_pts)
            print("Transformation matrix:")
            np.set_printoptions(precision=10, suppress=True)
            print(transform_matrix)

            # Scale the transformation matrix
            scale_matrix = np.array([
                [width, 0, 0],
                [0, height, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            transform_matrix = np.dot(scale_matrix, transform_matrix)
            print("Scaled transformation matrix:")
            print(transform_matrix)

            # Verify the transformation
            projected_pts = cv2.perspectiveTransform(np.array([src_pts]), transform_matrix)
            print(f"Projected points: {projected_pts}")

            return transform_matrix
        else:
            print("No circles detected or insufficient number of circles, retrying...")

        time.sleep(1)

# Function to move circles to align with the board corners
def move_circles_to_corners(screen, initial_positions, target_positions):
    # Calculate the differences between initial positions and target positions
    deltas = [(tx - ix, ty - iy) for (ix, iy), (tx, ty) in zip(initial_positions, target_positions)]

    while True:
        for i in range(len(initial_positions)):
            initial_positions[i] = (
                initial_positions[i][0] + deltas[i][0] // 10,
                initial_positions[i][1] + deltas[i][1] // 10
            )

        screen = project_circles(initial_positions)
        pygame.display.flip()

        if all(abs(tx - ix) < 10 and abs(ty - iy) < 10 for (ix, iy), (tx, ty) in zip(initial_positions, target_positions)):
            break

        time.sleep(0.5)

    return initial_positions

# Function to apply transformation and project corrected image
def apply_transform_and_project(transform_matrix):
    image = cv2.imread("example_image.png")
    if image is None:
        print("Failed to load image")
        return

    height, width = image.shape[:2]
    print("Original image dimensions:", width, height)
    print("Transform Matrix:")
    print(transform_matrix)

    # Apply the transformation and save the debug output
    corrected_image = cv2.warpPerspective(image, transform_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imwrite("corrected_image_original.png", corrected_image)

    # Inverted transformation for verification
    inverted_matrix = np.linalg.inv(transform_matrix)
    corrected_image_inv = cv2.warpPerspective(image, inverted_matrix, (width, height), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    cv2.imwrite("corrected_image_inverted.png", corrected_image_inv)

    # Convert the image to a format suitable for pygame
    corrected_image = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB)
    corrected_image = np.rot90(corrected_image)
    corrected_image = pygame.surfarray.make_surface(corrected_image)

    # Initialize pygame and display the image fullscreen
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen.blit(corrected_image, (0, 0))
    pygame.display.flip()

    # Keep displaying the image until the user quits
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                pygame.quit()

def main():
    if DEBUG_BOARD:
        # Simulate an average board in the middle of the screen
        height, width = 1080, 1920  # Assume 1080p resolution for the projector
        board_x, board_y, board_w, board_h = int(width * 0.25), int(height * 0.25), int(width * 0.5), int(height * 0.5)
        print(f"Simulating board at ({board_x}, {board_y}, {board_w}, {board_h})")

    # Initial positions of the circles (slightly inward from the corners)
    initial_positions = [(100, 100), (1820, 100), (100, 980), (1820, 980)]

    # Expected positions of the circles (board corners)
    target_positions = [(200, 200), (1720, 200), (200, 880), (1720, 880)]

    # Project initial circles
    screen = project_circles(initial_positions)

    # Wait a bit to ensure the projector is displaying the image
    time.sleep(2)

    # Move circles to align with the board corners
    final_positions = move_circles_to_corners(screen, initial_positions, target_positions)

    # Run the detection and adjustment in the main thread
    transform_matrix = detect_circles_and_calculate_transform(screen, final_positions)
    if transform_matrix is not None:
        apply_transform_and_project(transform_matrix)
    else:
        print("Failed to calculate transformation matrix")

if __name__ == "__main__":
    main()
