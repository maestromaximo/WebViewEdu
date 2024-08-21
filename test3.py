import cv2
import numpy as np
import pygame
import time
import qrcode
from PIL import Image

# Initialize QR Code detector
qr_detector = cv2.QRCodeDetector()

# Function to generate a QR code image
def generate_qr_code(data, size=100):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill='black', back_color='white')
    img = img.resize((size, size), Image.Resampling.LANCZOS)  # Updated to LANCZOS
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from PIL's RGB to OpenCV's BGR
    return img

# Function to project a QR code using pygame within a small debug board
def project_qr_code_on_debug_board(board_pos, board_size, qr_pos_within_board):
    height, width = 1080, 1920  # Assume 1080p resolution for the projector
    image = np.zeros((height, width, 3), np.uint8)
    
    # Generate the QR code image
    qr_code_image = generate_qr_code('Debug QR Code', size=100)
    
    # Calculate the position of the QR code within the debug board
    qr_x = board_pos[0] + qr_pos_within_board[0]
    qr_y = board_pos[1] + qr_pos_within_board[1]
    
    # Overlay the QR code image on the main image at the calculated position
    image[qr_y:qr_y+qr_code_image.shape[0], qr_x:qr_x+qr_code_image.shape[1]] = qr_code_image

    # Draw the debug board as a rectangle
    cv2.rectangle(image, board_pos, (board_pos[0] + board_size[0], board_pos[1] + board_size[1]), (255, 255, 255), 2)

    # Display the image using pygame
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rot90(image)
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    screen.blit(pygame.surfarray.make_surface(image), (0, 0))
    pygame.display.flip()

    return screen, (qr_x, qr_y)

# Function to detect the QR code and find its center
def detect_qr_code():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        retval, decoded_info, points, _ = qr_detector.detectAndDecodeMulti(frame)
        
        if points is not None:
            points = np.int32(points[0])
            center = np.mean(points, axis=0)
            return tuple(center)
        
    return None

# Function to create the mapping function
def create_mapping(projection_pos, detected_pos):
    dx = detected_pos[0] - projection_pos[0]
    dy = detected_pos[1] - projection_pos[1]
    
    def mapping_function(x, y):
        return x + dx, y + dy
    
    return mapping_function

# Function to project a test image within the mapped debug board area
def project_test_image(mapping_func, board_pos, board_size):
    height, width = 1080, 1920
    image = np.zeros((height, width, 3), np.uint8)
    
    # Define a test pattern (e.g., a grid or simple pattern)
    for i in range(0, board_size[0], 20):
        for j in range(0, board_size[1], 20):
            mapped_x, mapped_y = mapping_func(board_pos[0] + i, board_pos[1] + j)
            cv2.circle(image, (int(mapped_x), int(mapped_y)), 5, (0, 255, 0), -1)
    
    # Display the test image using pygame
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.rot90(image)
    screen = pygame.display.set_mode((1920, 1080), pygame.FULLSCREEN)
    screen.blit(pygame.surfarray.make_surface(image), (0, 0))
    pygame.display.flip()

def main():
    # Define the position and size of the debug board
    board_pos = (600, 400)
    board_size = (400, 300)
    qr_pos_within_board = (50, 50)  # QR code position within the debug board
    
    # Project a QR code on the debug board
    screen, qr_proj_pos = project_qr_code_on_debug_board(board_pos, board_size, qr_pos_within_board)
    
    # Wait for the projection to be visible
    time.sleep(2)
    
    # Detect the QR code and find its center
    detected_pos = detect_qr_code()
    
    if detected_pos:
        print(f"QR Code detected at: {detected_pos}")
        
        # Create the mapping function
        mapping_func = create_mapping(qr_proj_pos, detected_pos)
        
        # Project a test image within the debug board using the mapping
        project_test_image(mapping_func, board_pos, board_size)
    else:
        print("QR Code not detected.")

if __name__ == "__main__":
    main()
