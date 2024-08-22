import cv2
import numpy as np
import cv2.aruco as aruco
import pygame
import tempfile
import os
import time
from multiprocessing import Process, Queue, Array
from ctypes import c_float

# Initialize pygame for projection
pygame.init()
screen_info = pygame.display.Info()
screen_width, screen_height = screen_info.current_w, screen_info.current_h
screen = pygame.display.set_mode((screen_width, screen_height), pygame.FULLSCREEN)

# Create ArUco dictionary and parameters
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_1000)
detector_params = aruco.DetectorParameters()
detector_params.adaptiveThreshConstant = 7
detector_params.minMarkerPerimeterRate = 0.03
detector_params.maxMarkerPerimeterRate = 4.0
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Initialize the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

temp_dir = tempfile.gettempdir()

def generate_and_detect_markers(num_markers=20, marker_size=200):
    projector_points = []
    webcam_points = []

    marker_positions = []
    
    for i in range(num_markers):
        # Generate marker
        marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
        aruco.generateImageMarker(aruco_dict, i, marker_size, marker_img)
        
        # Save marker
        marker_path = os.path.join(temp_dir, f'aruco_marker_{i}.png')
        cv2.imwrite(marker_path, marker_img)
        
        # Display marker
        x = np.random.randint(0, screen_width - marker_size)
        y = np.random.randint(0, screen_height - marker_size)
        marker_positions.append((x, y))
        marker_surface = pygame.image.load(marker_path)
        screen.blit(marker_surface, (x, y))
        projector_points.append([x + marker_size / 2, y + marker_size / 2])

    pygame.display.flip()
    
    # Detect markers in a single frame
    ret, frame = cap.read()
    if not ret:
        return np.array(projector_points), np.array(webcam_points)

    corners, ids, _ = aruco_detector.detectMarkers(frame)
    
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id < num_markers:
                idx = np.where(ids == marker_id)[0][0]
                center = np.mean(corners[idx][0], axis=0)
                webcam_points.append(center)
    
    print(f"Detected {len(webcam_points)}/{num_markers} markers")
    return np.array(projector_points), np.array(webcam_points)

def calculate_homography(src_points, dst_points):
    if len(src_points) < 4 or len(dst_points) < 4:
        raise ValueError(f"Not enough points to calculate homography. Found {len(src_points)} points.")
    
    if src_points.shape != dst_points.shape:
        raise ValueError(f"Mismatch in point array shapes. src_points: {src_points.shape}, dst_points: {dst_points.shape}")
    
    # Use RANSAC for initial estimation
    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0, maxIters=2000)
    
    if H is None:
        raise ValueError("Failed to calculate homography with RANSAC")
    
    # Refine with Levenberg-Marquardt
    H, _ = cv2.findHomography(src_points, dst_points, cv2.LMEDS)
    
    return H

def translate_coordinates(point, H, inverse=False):
    if inverse:
        H = np.linalg.inv(H)
    point = np.array([point], dtype="float32")
    transformed_point = cv2.perspectiveTransform(np.array([point]), H)
    return transformed_point[0][0]

def detect_yellow_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Use morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    return None

def webcam_process(queue):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        yellow_center = detect_yellow_object(frame)
        if yellow_center:
            queue.put(yellow_center)

def main():
    print("Generating and detecting markers...")
    projector_points, webcam_points = generate_and_detect_markers()
    
    if len(projector_points) < 4 or len(webcam_points) < 4:
        print("Error: Not enough markers detected. Please ensure good lighting conditions and that markers are visible to the camera.")
        return
    
    print("Calculating homography...")
    try:
        H = calculate_homography(webcam_points, projector_points)
        print("Homography calculated successfully.")
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Start webcam process
    queue = Queue()
    p = Process(target=webcam_process, args=(queue,))
    p.start()
    
    try:
        while True:
            if not queue.empty():
                webcam_position = queue.get()
                projector_position = translate_coordinates(webcam_position, H)
                
                screen.fill((0, 0, 0))
                pygame.draw.circle(screen, (0, 255, 0), (int(projector_position[0]), int(projector_position[1])), 10)
                pygame.display.flip()
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                    raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("Exiting...")
    
    finally:
        p.terminate()
        cap.release()
        pygame.quit()
        cv2.destroyAllWindows()
        
        # Remove temporary files
        for i in range(20):
            marker_path = os.path.join(temp_dir, f'aruco_marker_{i}.png')
            if os.path.exists(marker_path):
                os.remove(marker_path)

if __name__ == "__main__":
    main()
