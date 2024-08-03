import cv2
import numpy as np
import time
import pygame
import os
import shutil

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

def generate_transformation_function(proj_points, cam_points):
    """
    Generates a transformation function to convert coordinates from webcam system to projector system.
    """
    assert len(proj_points) == 4 and len(cam_points) == 4, "Exactly four points are required."
    
    A = []
    B = []
    
    for (x_p, y_p), (x_w, y_w) in zip(proj_points, cam_points):
        A.append([x_w, y_w, 1, 0, 0, 0])
        A.append([0, 0, 0, x_w, y_w, 1])
        B.append(x_p)
        B.append(y_p)
    
    A = np.array(A)
    B = np.array(B)
    
    transform = np.linalg.lstsq(A, B, rcond=None)[0]
    
    def transform_coordinates(x_w, y_w):
        x_p = transform[0] * x_w + transform[1] * y_w + transform[2]
        y_p = transform[3] * x_w + transform[4] * y_w + transform[5]
        return x_p, y_p
    
    return transform_coordinates

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
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)  # Explicitly setting resolution to 1920x1080
    screen.blit(image, (0, 0))
    pygame.display.flip()

    return screen

# Function to detect circles and calculate the transformation
def detect_circles_and_calculate_transform(min_distance=50):
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

            # Filter out circles that are too close to each other
            filtered_positions = []
            for pos in detected_positions:
                if all(np.linalg.norm(np.array(pos) - np.array(existing_pos)) > min_distance for existing_pos in filtered_positions):
                    filtered_positions.append(pos)

            # If only three points are detected, estimate the fourth point
            if len(filtered_positions) == 3:
                p1, p2, p3 = filtered_positions
                missing_point = estimate_fourth_point(p1, p2, p3)
                filtered_positions.append(missing_point)

            if len(filtered_positions) == 4:
                print(f"Detected 4 positions: {filtered_positions}")
                # Draw green dots at the center of detected positions
                for (x, y) in filtered_positions:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
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

        # Order points to ensure consistency
        detected_positions = order_points(np.array(filtered_positions, dtype=np.float32))

        return detected_positions

def estimate_fourth_point(p1, p2, p3):
    # Calculate the centroid of the three points
    centroid = np.mean([p1, p2, p3], axis=0)
    # Find the fourth point as the reflection of the centroid over each pair of points
    for i in range(3):
        for j in range(i + 1, 3):
            if i != j:
                p4 = 2 * centroid - np.array(p1 if i == 0 else p2 if i == 1 else p3)
                if not any(np.array_equal(p4, p) for p in [p1, p2, p3]):
                    return tuple(p4)
    return p1  # Default, should not happen if three unique points are provided

# Function to apply transformation and project corrected image
def apply_transform_and_project(image_path, transform_func, cam_corners):
    # Load the image to be projected
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    # Calculate the projector coordinates for the given webcam corners
    proj_corners = np.array([transform_func(x, y) for x, y in cam_corners], dtype=np.float32)

    # Get the dimensions of the image
    height, width = image.shape[:2]
    print("Original image dimensions:", width, height)
    
    # Define the original image corners
    src_corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Calculate the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_corners, proj_corners)
    
    # Warp the image to fit the projector coordinates
    warped_image = cv2.warpPerspective(image, matrix, (1920, 1080))

    # Save the warped image for debugging
    cv2.imwrite("warped_image.png", warped_image)

    # Convert the image to a format suitable for pygame
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    warped_image = np.rot90(warped_image)
    warped_image = pygame.surfarray.make_surface(warped_image)

    # Initialize pygame and display the image fullscreen
    pygame.init()
    screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    screen.blit(warped_image, (0, 0))
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

    # Project initial circles
    screen = project_circles(initial_positions)

    # Wait a bit to ensure the projector is displaying the image
    time.sleep(2)

    # Detect circles and calculate the transformation
    detected_positions = detect_circles_and_calculate_transform()
    if len(detected_positions) == 4:
        # Generate the transformation function
        cam_points = [(int(x), int(y)) for x, y in detected_positions]
        proj_points = [(100, 100), (1820, 100), (100, 980), (1820, 980)]
        transform_func = generate_transformation_function(proj_points, cam_points)

        # Apply transformation and project the image
        cam_corners = [(200, 200), (1720, 200), (200, 880), (1720, 880)]  # Example corners in webcam coordinates

        # Capture image from webcam for debugging
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Place green dots on detected positions
            for (x, y) in cam_points:
                cv2.circle(frame, (int(x), int(y)), 10, (0, 255, 0), -1)  # Green dots

            # Place orange dots on cam corners
            for (x, y) in cam_corners:
                cv2.circle(frame, (int(x), int(y)), 10, (0, 165, 255), -1)  # Orange dots

            # Save the debug image
            debug_filename = 'draw_debug.png'
            cv2.imwrite(debug_filename, frame)
            print(f"Saved debug image: {debug_filename}")

        apply_transform_and_project("example_image.png", transform_func, cam_corners)
    else:
        print("Failed to calculate transformation matrix")

if __name__ == "__main__":
    main()
