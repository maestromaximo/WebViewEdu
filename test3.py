import cv2
import numpy as np
import cv2.aruco as aruco

# Initialize the camera
cap = cv2.VideoCapture(0)

# Generate an ArUco marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 42  # Any ID between 0 and 249
marker_size = 200  # Marker size in pixels
marker_img = aruco.drawMarker(aruco_dict, marker_id, marker_size)

# Save the marker image
cv2.imwrite('aruco_marker.png', marker_img)

# Project the marker using the projector (this part depends on your setup)
# Assuming you have a function to project the image (e.g., displaying it full screen on the projector)
# For simplicity, we assume the projector projects to the screen as is
# Display the image (you can customize this part for your projector)
cv2.imshow('Marker', marker_img)
cv2.waitKey(1000)  # Display for 1 second

# Detect the marker with the camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the ArUco marker
    corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict)
    
    # If marker detected
    if ids is not None and marker_id in ids:
        index = np.where(ids == marker_id)[0][0]
        marker_corners = corners[index][0]
        
        # Assume the first corner (top-left) as the detected position
        detected_position = np.mean(marker_corners, axis=0)
        print(f"Detected marker position: {detected_position}")
        
        # Let's assume the marker was projected at the center of the projector screen
        # Replace this with the actual projection coordinates
        projector_position = np.array([marker_size / 2, marker_size / 2])
        
        # Calculate the difference
        difference = detected_position - projector_position
        print(f"Difference between projected and detected: {difference}")
        
        break

# Define a function to translate coordinates
def translate_coordinates(point, to_projector=True):
    """
    Translates a point between camera and projector coordinate systems.
    
    :param point: The point to translate (x, y)
    :param to_projector: If True, translates from camera to projector; otherwise, reverse.
    :return: Translated point (x', y')
    """
    if to_projector:
        return point - difference
    else:
        return point + difference

# Example test of the function
camera_point = np.array([150, 150])
projector_point = translate_coordinates(camera_point, to_projector=True)
print(f"Camera point: {camera_point}, Projector point: {projector_point}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
