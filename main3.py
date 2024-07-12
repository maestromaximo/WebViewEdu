import cv2
import numpy as np

# Hyperparameters
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 2.5
DEBUG_VIEW = True
AREA_THRESHOLD = 1000
DISTANCE_THRESHOLD = 50  # Distance threshold to group writing areas

def detect_chalkboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4 and cv2.contourArea(approx) > AREA_THRESHOLD:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if ASPECT_RATIO_MIN < aspect_ratio < ASPECT_RATIO_MAX:
                return (x, y, w, h)
    return None

def detect_writing(frame, chalkboard):
    x, y, w, h = chalkboard
    chalkboard_roi = frame[y:y+h, x:x+w]
    gray_roi = cv2.cvtColor(chalkboard_roi, cv2.COLOR_BGR2GRAY)
    adaptive_thresh = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(morph, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > AREA_THRESHOLD:
            bx, by, bw, bh = cv2.boundingRect(contour)
            boxes.append((bx, by, bw, bh))
    
    return boxes

def group_boxes(boxes, distance_threshold):
    def overlap(box1, box2):
        return not (box1[0] + box1[2] < box2[0] - distance_threshold or
                    box1[0] - distance_threshold > box2[0] + box2[2] or
                    box1[1] + box1[3] < box2[1] - distance_threshold or
                    box1[1] - distance_threshold > box2[1] + box2[3])
    
    grouped_boxes = []
    while boxes:
        box = boxes.pop(0)
        group = [box]
        i = 0
        while i < len(boxes):
            if overlap(box, boxes[i]):
                group.append(boxes.pop(i))
            else:
                i += 1
        grouped_boxes.append(group)
    
    merged_boxes = []
    for group in grouped_boxes:
        x_min = min([b[0] for b in group])
        y_min = min([b[1] for b in group])
        x_max = max([b[0] + b[2] for b in group])
        y_max = max([b[1] + b[3] for b in group])
        merged_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    
    return merged_boxes

# Initialize the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    chalkboard = detect_chalkboard(frame)
    if chalkboard:
        x, y, w, h = chalkboard
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Chalkboard', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        writing_boxes = detect_writing(frame, chalkboard)
        grouped_boxes = group_boxes(writing_boxes, DISTANCE_THRESHOLD)
        
        for gx, gy, gw, gh in grouped_boxes:
            cv2.rectangle(frame[y:y+h, x:x+w], (gx, gy), (gx + gw, gy + gh), (255, 0, 0), 2)
    
    # Display the resulting frame
    cv2.imshow('Chalkboard and Writing Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
