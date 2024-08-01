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
AREA_THRESHOLD = 1000
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
DEBUG_BOARD = False  # Set this to True to simulate an average board
DEBUG_FOLDER_PHOTOS = True  # Set this to True to save processed images with detected circles
DEBUG_FOLDER_PATH = 'debug_photos'  # Path to the folder where debug photos will be saved

orange_count = 0

# Function to project circles at the corners
def project_circles():
    height, width = 1080, 1920  # Assume 1080p resolution for the projector
    image = np.zeros((height, width, 3), np.uint8)
    color = (255, 0, 255)  # Purple color in BGR
    radius = 50  # Slightly larger radius for better visibility
    thickness = -1  # Filled circles

    # Adjusted positions of the circles (moved inward from the corners)
    offset = 100
    positions = [(offset, offset), (width - offset, offset), (offset, height - offset), (width - offset, height - offset)]
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

    # Keep displaying the image until the user quits
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                pygame.quit()
                return

# Function to detect circles and calculate the transformation
def detect_circles_and_calculate_transform():
    image_counter = 0
    while True:
        # Capture image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture image from webcam")
            continue

        print("Captured image from webcam")

        # Convert image to grayscale and apply GaussianBlur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (11, 11), 0)
        print("Converted image to grayscale and applied GaussianBlur")

        # Detect circles using HoughCircles
        circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100, param1=50, param2=30, minRadius=30, maxRadius=70)

        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            detected_positions = [(x, y) for (x, y, r) in circles]
            print(f"Detected circles at positions: {detected_positions}")

            # Draw green dots at the center of detected circles
            for (x, y) in detected_positions:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

            # Save the processed image if DEBUG_FOLDER_PHOTOS is True
            if DEBUG_FOLDER_PHOTOS:
                if not os.path.exists(DEBUG_FOLDER_PATH):
                    os.makedirs(DEBUG_FOLDER_PATH)
                else:
                    for filename in os.listdir(DEBUG_FOLDER_PATH):
                        file_path = os.path.join(DEBUG_FOLDER_PATH, filename)
                        os.remove(file_path)
                debug_filename = os.path.join(DEBUG_FOLDER_PATH, f"debug_frame_{image_counter}.png")
                cv2.imwrite(debug_filename, frame)
                print(f"Saved debug frame: {debug_filename}")
                image_counter += 1

            # Expected positions of the circles (adjusted inward)
            height, width = frame.shape[:2]
            offset = 100
            expected_positions = [(offset, offset), (width - offset, offset), (offset, height - offset), (width - offset, height - offset)]

            # Calculate transformation
            if len(detected_positions) == 4:
                src_pts = np.array(detected_positions, dtype=np.float32)
                dst_pts = np.array(expected_positions, dtype=np.float32)
                transform_matrix, _ = cv2.findHomography(src_pts, dst_pts)
                print("Calculated transformation matrix")
                return transform_matrix
            else:
                print(f"Expected 4 circles but detected {len(detected_positions)}, retrying...")
        else:
            print("No circles detected, retrying...")

        time.sleep(1)

# Function to apply transformation and project corrected image
def apply_transform_and_project(transform_matrix):
    # Load an example image to project
    image = cv2.imread("example_image.png")
    if image is None:
        print("Failed to load image")
        return

    # Apply the transformation
    height, width = image.shape[:2]
    corrected_image = cv2.warpPerspective(image, transform_matrix, (width, height))

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

    # Create a thread for projecting the circles
    projection_thread = threading.Thread(target=project_circles)
    projection_thread.start()

    # Wait a bit to ensure the projector is displaying the image
    time.sleep(2)

    # Run the detection and adjustment in the main thread
    transform_matrix = detect_circles_and_calculate_transform()
    if transform_matrix is not None:
        apply_transform_and_project(transform_matrix)
    else:
        print("Failed to calculate transformation matrix")

if __name__ == "__main__":
    main()
