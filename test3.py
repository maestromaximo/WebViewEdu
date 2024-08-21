import cv2
import numpy as np
import pygame
import qrcode
from PIL import Image

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

    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    img = np.array(img, dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert from PIL's RGB to OpenCV's BGR

    return img

# Function to project a QR code using pygame within a small debug board
def project_qr_code_on_debug_board(board_pos, board_size, qr_pos_within_board):
    pygame.init()
    height, width = 1080, 1920  # Assume 1080p resolution for the projector
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    surface = pygame.Surface((width, height))
    qr_code_image = generate_qr_code('Debug QR Code', size=100)

    # Calculate the position of the QR code within the debug board
    qr_x = board_pos[0] + qr_pos_within_board[0]
    qr_y = board_pos[1] + qr_pos_within_board[1]
    
    # Blit the QR code image onto the surface
    surface.blit(pygame.surfarray.make_surface(qr_code_image), (qr_x, qr_y))
    
    # Draw the debug board as a rectangle
    pygame.draw.rect(surface, (255, 255, 255), (board_pos[0], board_pos[1], board_size[0], board_size[1]), 2)
    
    # Display the image using pygame
    screen.blit(surface, (0, 0))
    pygame.display.flip()

    return screen, (qr_x, qr_y)

def main():
    # Define the position and size of the debug board
    board_pos = (600, 400)
    board_size = (400, 300)
    qr_pos_within_board = (50, 50)  # QR code position within the debug board
    
    # Project a QR code on the debug board
    screen, qr_proj_pos = project_qr_code_on_debug_board(board_pos, board_size, qr_pos_within_board)
    
    # Keep the window open until it's closed by the user
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
    pygame.quit()

if __name__ == "__main__":
    main()
