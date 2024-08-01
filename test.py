import cv2
import numpy as np
import time
import os

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
DEBUG_BOARD = True  # Set this to True to simulate an average board
orange_count = 0

# Function to project purple circles at the corners
def project_purple_circles():
    height, width = 1080, 1920  # Assume 1080p resolution for the projector
    image = np.zeros((height, width, 3), np.uint8)
    color = (255, 0, 255)  # Purple color in BGR
    radius = 30
    thickness = -1  # Filled circles

    # Positions of the circles
    positions = [(radius, radius), (width - radius, radius), (radius, height - radius), (width - radius, height - radius)]
    for (x, y) in positions:
        cv2.circle(image, (x, y), radius, color, thickness)

    # Project the image (for now, just save it to a file for debugging)
    cv2.imwrite("projected_circles.png", image)
    # Code to actually project the image using your projector setup would go here

# Function to detect purple circles and calculate the transformation
def detect_purple_circles_and_calculate_transform():
    while True:
        # Capture image from webcam
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print("Failed to capture image from webcam")
            continue

        # Convert image to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_purple = np.array([130, 50, 50])
        upper_purple = np.array([160, 255, 255])
        mask = cv2.inRange(hsv, lower_purple, upper_purple)

        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        detected_positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Minimum area to filter noise
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    detected_positions.append((cx, cy))

        # Expected positions of the circles
        height, width = frame.shape[:2]
        radius = 30
        expected_positions = [(radius, radius), (width - radius, radius), (radius, height - radius), (width - radius, height - radius)]

        # Calculate transformation
        if len(detected_positions) == 4:
            src_pts = np.array(detected_positions, dtype=np.float32)
            dst_pts = np.array(expected_positions, dtype=np.float32)
            transform_matrix, _ = cv2.findHomography(src_pts, dst_pts)
            return transform_matrix
        else:
            print("Failed to detect all circles, retrying...")
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

    # Project the corrected image (for now, just save it to a file for debugging)
    cv2.imwrite("corrected_image.png", corrected_image)
    # Code to actually project the image using your projector setup would go here

def main():
    if DEBUG_BOARD:
        # Simulate an average board in the middle of the screen
        height, width = 1080, 1920  # Assume 1080p resolution for the projector
        board_x, board_y, board_w, board_h = int(width * 0.25), int(height * 0.25), int(width * 0.5), int(height * 0.5)
        print(f"Simulating board at ({board_x}, {board_y}, {board_w}, {board_h})")

    project_purple_circles()
    time.sleep(1)  # Wait for the projector to display the image
    transform_matrix = detect_purple_circles_and_calculate_transform()
    if transform_matrix is not None:
        apply_transform_and_project(transform_matrix)
    else:
        print("Failed to calculate transformation matrix")

if __name__ == "__main__":
    main()
