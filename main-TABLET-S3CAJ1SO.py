import cv2
import numpy as np
import time
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import requests
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')

# Hyperparameters
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 2.5
DEBUG_VIEW = True
DEBUG_FEED = True  # New hyperparameter for using video file
VIDEO_PATH = 'C:/Users/aleja/OneDrive/Escritorio/WebViewEdu/test_vid1.mp4'  # Path to the video file
PERSON_DETECTION = False  # Hyperparameter to control person detection
AREA_THRESHOLD = 1000
DISTANCE_THRESHOLD = 1  # Adjusted based on the requirement
STABILITY_THRESHOLD = 10  # Pixels threshold for movement
CONFIRMATION_TIME = 10  # Seconds to confirm chalkboard stability
REAPPEARANCE_TIME = 0.5  # Seconds to allow brief periods without detection
CUSHION = 10  # Pixels to reduce the chalkboard's bounding box for average color calculation
WRITING_CHECK_INTERVAL = 5  # Seconds interval to recheck for new writing
TEXT_DETECTION_URL = "https://aleale2423-textdetector.hf.space/detect"
MARGIN = 10  # Extra pixels for the aggregated bounding box

# Initialize YOLOS model for person detection if enabled
if PERSON_DETECTION:
    person_model = YolosForObjectDetection.from_pretrained('hustvl/yolos-tiny')
    image_processor = YolosImageProcessor.from_pretrained('hustvl/yolos-tiny')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    person_model.to(device)

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
                return (x, y, w, h), approx
    return None, None

def calculate_average_color(frame, chalkboard):
    x, y, w, h = chalkboard
    cushioned_x = x + CUSHION
    cushioned_y = y + CUSHION
    cushioned_w = w - 2 * CUSHION
    cushioned_h = h - 2 * CUSHION
    roi = frame[cushioned_y:cushioned_y + cushioned_h, cushioned_x:cushioned_x + cushioned_w]
    average_color = cv2.mean(roi)[:3]
    return average_color

def detect_person(frame):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    inputs = image_processor(images=pil_image, return_tensors='pt').to(device)
    outputs = person_model(**inputs)

    target_sizes = torch.tensor([pil_image.size[::-1]], device=device)
    results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    
    for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
        if person_model.config.id2label[label.item()] == 'person':
            box = [round(i, 2) for i in box.tolist()]
            return int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])
    return None

def send_to_text_detection_service(image):
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    
    headers = {
        'Authorization': f'Bearer {HF_TOKEN}'
    }
    files = {
        'file': ('image.png', img_bytes, 'image/png')
    }
    
    try:
        response = requests.post(TEXT_DETECTION_URL, headers=headers, files=files)
        response.raise_for_status()
        return response.json().get("boxes", [])
    except requests.exceptions.RequestException as e:
        print("Error in text detection service:", e)
        return []

def merge_boxes(boxes, margin, x_max, y_max):
    if not boxes:
        return []
    
    x_min = min([b[0] for b in boxes]) - margin
    y_min = min([b[1] for b in boxes]) - margin
    x_max_box = max([b[0] + b[2] for b in boxes]) + margin
    y_max_box = max([b[1] + b[3] for b in boxes]) + margin

    x_min = max(x_min, 0)
    y_min = max(y_min, 0)
    x_max_box = min(x_max_box, x_max)
    y_max_box = min(y_max_box, y_max)

    return [(x_min, y_min, x_max_box - x_min, y_max_box - y_min)]

# Initialize the webcam or video feed
if DEBUG_FEED:
    cap = cv2.VideoCapture(VIDEO_PATH)
else:
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

chalkboard_coords = None
chalkboard_corners = None
chalkboard_confirmed = False
start_time = None
last_detection_time = None
average_color = None
writing_boxes = []
last_writing_check = 0

while True:
    ret, frame = cap.read()
    if not ret:
        if DEBUG_FEED:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
            continue
        else:
            print("Failed to grab frame.")
            break

    if not chalkboard_confirmed:
        detected_chalkboard, detected_corners = detect_chalkboard(frame)
        if detected_chalkboard:
            x, y, w, h = detected_chalkboard
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Detecting Chalkboard', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print("Chalkboard detected, verifying stability...")

            if chalkboard_coords:
                prev_x, prev_y, prev_w, prev_h = chalkboard_coords
                if (abs(prev_x - x) < STABILITY_THRESHOLD and 
                    abs(prev_y - y) < STABILITY_THRESHOLD and 
                    abs(prev_w - w) < STABILITY_THRESHOLD and 
                    abs(prev_h - h) < STABILITY_THRESHOLD):
                    if not start_time:
                        start_time = time.time()
                    elif time.time() - start_time > CONFIRMATION_TIME:
                        chalkboard_confirmed = True
                        chalkboard_coords = detected_chalkboard
                        chalkboard_corners = detected_corners
                        average_color = calculate_average_color(frame, chalkboard_coords)
                        print(f'Chalkboard confirmed at {chalkboard_coords}')
                else:
                    start_time = None
                last_detection_time = time.time()
            else:
                chalkboard_coords = detected_chalkboard
                chalkboard_corners = detected_corners
                start_time = time.time()
                last_detection_time = time.time()
        else:
            if last_detection_time and time.time() - last_detection_time < REAPPEARANCE_TIME:
                print("Chalkboard temporarily lost, waiting for reappearance...")
            else:
                start_time = None

    if chalkboard_confirmed:
        x, y, w, h = chalkboard_coords
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Chalkboard', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw stability threshold circles on corners
        for corner in chalkboard_corners:
            cx, cy = corner[0][0], corner[0][1]
            cv2.circle(frame, (cx, cy), STABILITY_THRESHOLD, (0, 255, 255), 2)
        
        # Detect writing within the chalkboard
        current_time = time.time()
        if current_time - last_writing_check > WRITING_CHECK_INTERVAL:
            chalkboard_roi = frame[y:y+h, x:x+w]
            detected_boxes = send_to_text_detection_service(chalkboard_roi)
            writing_boxes = [(int(box['startX']), int(box['startY']), int(box['endX']) - int(box['startX']), int(box['endY']) - int(box['startY'])) for box in detected_boxes] if detected_boxes else []
            last_writing_check = current_time
        
        if writing_boxes:
            merged_box = merge_boxes(writing_boxes, margin=MARGIN, x_max=w, y_max=h)
        else:
            merged_box = []
        
        for gx, gy, gw, gh in merged_box:
            cv2.rectangle(frame[y:y+h, x:x+w], (gx, gy), (gx + gw, gy + gh), (255, 0, 0), 2)
            if DEBUG_VIEW:
                cv2.putText(frame, 'Writing', (gx, gy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        if PERSON_DETECTION:
            person_box = detect_person(frame)
            if person_box:
                px, py, pw, ph = person_box
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
                cv2.putText(frame, 'Person', (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the resulting frame
    if DEBUG_VIEW:
        cv2.imshow('Chalkboard and Writing Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
