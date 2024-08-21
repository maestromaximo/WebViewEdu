import cv2
import numpy as np
import pygame
import time
import os

# Hyperparameters
DEBUG_VIEW = False
VIDEO_PATH = 'C:/Users/aleja/OneDrive/Escritorio/WebViewEdu/test_vid1.mp4'
AREA_THRESHOLD = 10
DEBUG_FOLDER_PATH = 'debug_photos'

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
    def transform_coordinates(x, y):
        point = np.array([[x, y]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(np.array([point]), matrix)
        return transformed_point[0][0]
    return transform_coordinates

def project_circles(positions):
    height, width = 1080, 1920
    image = np.zeros((height, width, 3), np.uint8)
    color = (255, 0, 255)
    radius = 50

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

    return detected_positions

def apply_transform_and_project(transform_func, image_path="example_image.png"):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image")
        return

    height, width = image.shape[:2]
    src_corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

    proj_corners = np.array([transform_func(x, y) for x, y in src_corners], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_corners, proj_corners)
    warped_image = cv2.warpPerspective(image, matrix, (1920, 1080))

    pygame.init()
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    warped_image = cv2.cvtColor(warped_image, cv2.COLOR_BGR2RGB)
    warped_image = np.rot90(warped_image)
    warped_image = pygame.surfarray.make_surface(warped_image)
    screen.blit(warped_image, (0, 0))
    pygame.display.flip()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                pygame.quit()

def main():
    # Project initial circles
    initial_positions = [(100, 100), (1820, 100), (100, 980), (1820, 980)]
    project_circles(initial_positions)
    time.sleep(2)

    detected_positions = detect_circles()
    if len(detected_positions) == 4:
        detected_positions = order_points(np.array(detected_positions, dtype=np.int32))
        transform_func = generate_transformation_function(initial_positions, detected_positions)
        apply_transform_and_project(transform_func)
    else:
        print("Failed to detect 4 circles for transformation")

if __name__ == "__main__":
    main()
