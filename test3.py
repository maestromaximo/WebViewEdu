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
        return None

    # Save the frame for debugging
    if debug:
        cv2.imwrite('debug_frame.jpg', frame)

    decoded_info, points, _ = qr_detector.detectAndDecode(frame)
    if points is not None:
        points = np.int32(points)
        center = np.mean(points, axis=0)
        return center
    else:
        print("QR Code not detected. Check debug_frame.jpg for analysis.")
        return None


# Function to create the mapping function
def create_mapping(projection_pos, detected_pos):
    dx, dy = detected_pos - projection_pos
    def mapping_function(x, y):
        return int(x + dx), int(y + dy)
    print(f"Mapping function created with dx={dx}, dy={dy}.")
    return mapping_function

# Function to project an image using a mapping function
def project_image(screen, mapping_func, image_path):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image.")
        return

    # Assuming the image should be mapped to the entire screen for simplicity
    img_height, img_width = img.shape[:2]
    img_corners = np.float32([[0, 0], [img_width, 0], [img_width, img_height], [0, img_height]])
    projected_corners = np.float32([mapping_func(x, y) for x, y in img_corners])

    matrix = cv2.getPerspectiveTransform(img_corners, projected_corners)
    transformed_img = cv2.warpPerspective(img, matrix, (1920, 1080))
    transformed_surface = pygame.surfarray.make_surface(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    screen.blit(transformed_surface, (0, 0))
    pygame.display.flip()
    print("Image projected using mapping.")

def generate_and_project_qr(screen, board_pos, qr_code_size=600):  # Default size set to 600
    # Initialize QR code with high error correction and specified dimensions
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,
        border=4,
    )
    qr.add_data('Debug QR Code')
    qr.make(fit=True)

    # Convert QR code to an image
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    img = img.resize((qr_code_size, qr_code_size), Image.Resampling.LANCZOS)  # Ensure qr_code_size is an integer
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Determine top-left corner of the QR code on the screen
    qr_x = board_pos[0]
    qr_y = board_pos[1]

    # Convert image for Pygame and blit it onto the screen
    img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    screen.blit(img_surface, (qr_x, qr_y))
    pygame.display.flip()  # Update the display

    return (qr_x, qr_y), img



def main():
    pygame.init()
    screen = pygame.display.set_mode((1920, 1080))

    board_pos = (300, 200)  # Top-left corner of the QR code
    qr_code_size = 600      # Size of the QR code to generate

    qr_position, qr_img = generate_and_project_qr(screen, board_pos, qr_code_size)
    
    
    time.sleep(4)

    detected_center = detect_qr_code(debug=True)

    if detected_center:
        print(f"Detected QR Code at: {detected_center}")
    else:
        print("Failed to detect QR Code.")

    pygame.quit()

if __name__ == "__main__":
    main()