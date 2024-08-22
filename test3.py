import cv2
import numpy as np
import cv2.aruco as aruco

# Initialize the camera
cap = cv2.VideoCapture(0)

# Create ArUco dictionary and generate the marker
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
marker_id = 42
marker_size = 200  # Marker size in pixels
marker_img = np.zeros((marker_size, marker_size, 1), dtype="uint8")
aruco.generateImageMarker(aruco_dict, marker_id, marker_size, marker_img)

# Save the marker image
cv2.imwrite('aruco_marker.png', marker_img)

# Display the marker (assumes you have a method to project this)
cv2.imshow('Marker', marker_img)
cv2.waitKey(1000)  # Display for 1 second

# Initialize ArUco Detector
detector_params = aruco.DetectorParameters()
aruco_detector = aruco.ArucoDetector(aruco_dict, detector_params)

# Detect the marker with the camera
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect markers
    corners, ids, rejected = aruco_detector.detectMarkers(frame)
    
    # If marker detected
    if ids is not None and marker_id in ids:
        index = np.where(ids == marker_id)[0][0]
        marker_corners = corners[index][0]
        
        # Assume the first corner (top-left) as the detected position
        detected_position = np.mean(marker_corners, axis=0)
        print(f"Detected marker position: {detected_position}")
        
        # Assume marker was projected at the center of the screen
        projector_position = np.array([marker_size / 2, marker_size / 2])
        
        # Calculate the difference
        difference = detected_position - projector_position
        print(f"Difference between projected and detected: {difference}")
        
        break

# Function to translate coordinates between camera and projector
def translate_coordinates(point, to_projector=True):
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
