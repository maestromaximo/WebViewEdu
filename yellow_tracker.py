import cv2
import numpy as np

def detect_yellow_center(frame):
    # Convert the frame to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the range for yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # Create a binary mask where yellow colors are white and the rest are black
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour, assuming it's the main yellow object
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the moments to calculate the centroid
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            return (cX, cY)
    
    return None  # Return None if no yellow object is detected

def main():
    # Initialize the video capture (0 for default camera)
    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        # Detect yellow center in the frame
        yellow_center = detect_yellow_center(frame)

        if yellow_center is not None:
            print(f"Yellow object detected at: {yellow_center}")
        else:
            print("No yellow object detected")

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture
    cap.release()

if __name__ == "__main__":
    main()
