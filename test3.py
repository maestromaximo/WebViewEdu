import cv2
import numpy as np
import pygame
import qrcode
from PIL import Image
import os
import shutil
import time

# Ensure debug_photos directory is clean
debug_photos = 'debug_photos'
if os.path.exists(debug_photos):
    shutil.rmtree(debug_photos)
os.makedirs(debug_photos)

# Initialize QR Code detector
qr_detector = cv2.QRCodeDetector()

# Function to detect the QR code
def detect_qr_code(debug=False):
    cap = cv2.VideoCapture(0)  # Adjust the device index if needed
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Failed to capture video frame")
        if debug:
            cv2.imwrite('debug_frame.jpg', frame)
        return None

    decoded_info, points, _ = qr_detector.detectAndDecode(frame)
    if points is not None:
        points = np.int32(points)
        center = np.mean(points, axis=0)
        if debug:
            cv2.imwrite('debug_frame.jpg', frame)
        return points.reshape(-1, 2).tolist()  # Reshape and convert to list for easier handling
    else:
        if debug:
            cv2.imwrite('debug_frame.jpg', frame)
        print("QR Code not detected. Check debug_frame.jpg for analysis.")
        return None

# Function to create the mapping function
def create_mapping(projection_pos, detected_corners):
    qr_code_size = 600  # Assuming QR code size is 600x600
    projected_corners = np.float32([
        [projection_pos[0], projection_pos[1]], 
        [projection_pos[0] + qr_code_size, projection_pos[1]],
        [projection_pos[0] + qr_code_size, projection_pos[1] + qr_code_size],
        [projection_pos[0], projection_pos[1] + qr_code_size]
    ])
    matrix = cv2.getPerspectiveTransform(np.array(detected_corners, dtype=np.float32), projected_corners)
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
    print("Image projected using mapping.")

# Function to generate and project QR code
def generate_and_project_qr(screen, board_pos, qr_code_size=600):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    qr.add_data('Debug QR Code')
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    img = img.resize((qr_code_size, qr_code_size), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    qr_x, qr_y = board_pos
    img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    screen.blit(img_surface, (qr_x, qr_y))
    pygame.display.flip()
    return (qr_x, qr_y)

def main():
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))
    board_pos = (300, 200)
    qr_position = generate_and_project_qr(screen, board_pos)

    time.sleep(4)  # Allow time for the QR code to be stable on screen

    detected_corners = detect_qr_code(debug=True)
    if detected_corners:
        print(f"Detected QR Code at: {detected_corners}")
        transform_matrix = create_mapping(qr_position, detected_corners)
        project_image(screen, transform_matrix, 'example_image.png')
    else:
        print("Failed to detect QR Code.")

    # Keep the window open until closed by the user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()
