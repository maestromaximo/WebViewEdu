import cv2
import numpy as np
import time
import pygame
import os
import shutil

# Hyperparameters
DEBUG_FEED = True
AREA_THRESHOLD = 10
DEBUG_FOLDER_PATH = 'debug_photos'
DEBUG_BOARD = True  # Set this to True to simulate an average board

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def generate_transformation_function(proj_points, cam_points):
    matrix = cv2.getPerspectiveTransform(np.array(cam_points, dtype=np.float32), np.array(proj_points, dtype=np.float32))
    return matrix

def project_circles(positions):
    height, width = 1080, 1920  # Assume 1080p resolution for the projector
    image = np.zeros((height, width, 3), np.uint8)
    color = (255, 0, 255)  # Purple color in BGR
    radius = 50  # Slightly larger radius for better visibility

    for (x, y) in positions:
        cv2.circle(image, (x, y), radius, color, -1)

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rot90(image)
    image = pygame.surfarray.make_surface(image)
    screen.blit(image, (0, 0))
    pygame.display.flip()

    return screen

def detect_circles():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture image from webcam")
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_positions = []
    for contour in contours:
        if cv2.contourArea(contour) > AREA_THRESHOLD:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                detected_positions.append((cX, cY))

    if len(detected_positions) == 4:
        detected_positions = order_points(np.array(detected_positions, dtype=np.int32))

    return detected_positions

def apply_transform_and_project(transform_matrix, cam_corners, image_path="example_image.png"):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    height, width = image.shape[:2]
    src_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    
    proj_corners = np.array([cv2.perspectiveTransform(np.array([[pt]], dtype=np.float32), transform_matrix)[0][0] for pt in cam_corners], dtype=np.float32)
    
    matrix = cv2.getPerspectiveTransform(src_corners, proj_corners)
    warped_image = cv2.warpPerspective(image, matrix, (1920, 1080))

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    warped_image = np.rot90(warped_image)
    warped_image = pygame.surfarray.make_surface(warped_image)
    screen.blit(warped_image, (0, 0))
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                pygame.quit()
                return

def main():
    if DEBUG_BOARD:
        print("Simulating board in the middle of the screen")

    initial_positions = [(100, 100), (1820, 100), (100, 980), (1820, 980)]
    project_circles(initial_positions)
    time.sleep(2)

    detected_positions = detect_circles()
    if len(detected_positions) == 4:
        transform_matrix = generate_transformation_function(initial_positions, detected_positions)
        cam_corners = [(200, 200), (1720, 200), (200, 880), (1720, 880)]
        apply_transform_and_project(transform_matrix, cam_corners)
    else:
        print("Failed to detect 4 circles for transformation")

if __name__ == "__main__":
    main()
