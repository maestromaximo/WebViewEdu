import cv2
import numpy as np
import pygame
import qrcode
from PIL import Image

# Initialize QR Code detector
qr_detector = cv2.QRCodeDetector()

# Function to generate a QR code image
def generate_qr_code(data, size=300):  # Increased size for better detection
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img

# Function to project a QR code using pygame within a small debug board
def project_qr_code_on_debug_board(board_pos, board_size, qr_pos_within_board):
    pygame.init()
    height, width = 1080, 1920  # Assume 1080p resolution for the projector
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    surface = pygame.Surface((width, height))
    qr_code_image = generate_qr_code('Debug QR Code', size=300)  # Larger QR code

    qr_x = board_pos[0] + qr_pos_within_board[0]
    qr_y = board_pos[1] + qr_pos_within_board[1]

    surface.blit(pygame.surfarray.make_surface(qr_code_image), (qr_x, qr_y))
    pygame.draw.rect(surface, (255, 255, 255), (board_pos[0], board_pos[1], board_size[0], board_size[1]), 2)

    screen.blit(surface, (0, 0))
    pygame.display.flip()

    return screen, (qr_x, qr_y)

# Function to detect the QR code and find its center
def detect_qr_code():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if ret:
        decoded_info, points, straight_qrcode = qr_detector.detectAndDecode(frame)
        if points is not None:
            points = np.int32(points)
            center = tuple(np.mean(points, axis=0))
            return center
    return None

# Function to create the mapping function
def create_mapping(projection_pos, detected_pos):
    dx = detected_pos[0] - projection_pos[0]
    dy = detected_pos[1] - projection_pos[1]

    def mapping_function(x, y):
        return int(x + dx), int(y + dy)

    return mapping_function

def main():
    board_pos = (300, 200)
    board_size = (600, 600)  # Adjusted to fit larger QR code
    qr_pos_within_board = (150, 150)  # Center the QR code within the board

    screen, qr_proj_pos = project_qr_code_on_debug_board(board_pos, board_size, qr_pos_within_board)

    # Wait a bit to ensure the QR code is visible and then detect
    pygame.time.wait(2000)
    detected_pos = detect_qr_code()

    if detected_pos:
        print(f"QR Code detected at: {detected_pos}")
        mapping_func = create_mapping(qr_proj_pos, detected_pos)
        # Here you can use mapping_func to project any image or adjust projections
        print(f"Mapping function created: {mapping_func}")
    else:
        print("QR Code not detected.")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

    pygame.quit()

if __name__ == "__main__":
    main()
