import json
import tempfile
import cv2
import numpy as np
import time
import torch
from transformers import YolosImageProcessor, YolosForObjectDetection
from PIL import Image
import requests
import os
from io import BytesIO
import base64
import openai
from dotenv import load_dotenv
from autogen import ConversableAgent, AssistantAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Load environment variables
load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
openai.api_key = OPENAI_API_KEY

# Hyperparameters
ASPECT_RATIO_MIN = 1.5
ASPECT_RATIO_MAX = 2.5
DEBUG_VIEW = True
DEBUG_FEED = True  # New hyperparameter for using video file
VIDEO_PATH = 'C:/Users/aleja/OneDrive/Escritorio/WebViewEdu/test_vid1.mp4'  # Path to the video file
PERSON_DETECTION = False  # Hyperparameter to control person detection
AREA_THRESHOLD = 1000
DISTANCE_THRESHOLD = 1  # Adjusted based on the requirement
STABILITY_THRESHOLD = 30  # Pixels threshold for movement
CONFIRMATION_TIME = 10  # Seconds to confirm chalkboard stability
REAPPEARANCE_TIME = 0.5  # Seconds to allow brief periods without detection
CUSHION = 10  # Pixels to reduce the chalkboard's bounding box for average color calculation
WRITING_CHECK_INTERVAL = 5  # Seconds interval to recheck for new writing
TEXT_DETECTION_URL = "https://aleale2423-textdetector.hf.space/detect"
MARGIN = 10  # Extra pixels for the aggregated bounding box
AREA_CHANGE_THRESHOLD = 0.2  # Minimum percentage change in area to trigger LaTeX conversion (20%)
DETAIL_LEVEL = "low"  # Options for detail level: low, high, auto
ORANGE_THRESHOLD = 50  # Pixels radius to ignore further orange circles

orange_count = 0

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

def analyze_image_with_question(image, question):
    """
    Send an image to OpenAI Vision API and ask a question about it.

    Args:
        image (numpy array): The image data.
        question (str): The question to ask about the image.

    Returns:
        str: The answer to the question about the image.
    """
    
    # Encode the image in base64
    _, img_encoded = cv2.imencode('.png', image)
    img_bytes = img_encoded.tobytes()
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    img_data_uri = f"data:image/png;base64,{img_base64}"

    # Create the payload with the image and question
    payload = {
        "model": "gpt-4o",
        "response_format": { "type": "json_object" },
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question,
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": img_data_uri,
                            "detail": DETAIL_LEVEL,
                        },
                    },
                ],
            }
        ]
    }
    
    # Send the payload to the API
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"
    }
    
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    l_dict = response.json()['choices'][0]['message']['content']
    l_dict = json.loads(l_dict)
    if isinstance(l_dict, dict) and len(list(l_dict.values())) > 0:
        return list(l_dict.values())[0]
    else:
        return "no text detected"

def detect_orange_circle(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_orange = np.array([5, 150, 150])
    upper_orange = np.array([15, 255, 255])
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Threshold to avoid detecting small dots
            ((x, y), radius) = cv2.minEnclosingCircle(contour)
            if radius > 5:  # Minimum radius to consider it a circle
                return int(x), int(y), int(radius)
    return None, None, None

def run_code_task(prompt):
    # Load environment variables
    load_dotenv()
    
    # Replace with your actual API key
    api_key = os.getenv("OPENAI_API_KEY")

    # Directory to store code files
    coding_dir = "coding"
    if not os.path.exists(coding_dir):
        os.makedirs(coding_dir)

    # Clean the temporary directory
    temp_dir = tempfile.TemporaryDirectory(dir=coding_dir)

    # Create a local command line code executor
    executor = LocalCommandLineCodeExecutor(
        timeout=10,  # Timeout for each code execution in seconds.
        work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
    )

    # Create the code executor agent
    code_executor_agent = ConversableAgent(
        name="CodeExecutorAgent",
        system_message="You execute code provided to you.",
        llm_config=False,  # Turn off LLM for this agent.
        code_execution_config={"executor": executor},  # Use the local command line code executor.
        human_input_mode="NEVER",  # Always take human input for this agent for safety.
    )

    # Create the code writer agent
    code_writer_agent = AssistantAgent(
        name="CodeWriterAgent",
        llm_config={"config_list": [{"model": "gpt-4", "api_key": api_key}]},
        code_execution_config=False,  # Turn off code execution for this agent.
    )

    def execute_task(prompt):
        # Clean the temporary directory at the beginning
        for file in os.listdir(temp_dir.name):
            file_path = os.path.join(temp_dir.name, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        # Start the conversation with the initial prompt
        chat_result = code_executor_agent.initiate_chat(
            recipient=code_writer_agent,
            message=f"Identify if this problem needs to be graphed, computed, or solved: {prompt}. "
                    "If it needs to be graphed, write a Python script to graph it and save it as 'plot.png'. "
                    "Do not display the plot. "
                    "If it needs to be computed or solved, write a Python script to compute or solve it, and save the result in 'result.txt'. "
                    "Terminate after performing the required action."
                    "The problem may contain mathematical expressions, equations, or functions, and may contain information outside the problem, identify what is relevant and then proceed with the task.",
            max_turns=5,  # Adjust the number of turns as needed
            summary_method="last_msg"
        )

        # Print the conversation history
        for message in chat_result.chat_history:
            print(f"{message['role']}: {message['content']}")

        # Check for plot.png or result.txt
        result = None
        if 'plot.png' in os.listdir(temp_dir.name):
            result = os.path.join(temp_dir.name, 'plot.png')
        elif 'result.txt' in os.listdir(temp_dir.name):
            with open(os.path.join(temp_dir.name, 'result.txt'), 'r') as file:
                result = file.read()
        
        # Clean up the temporary directory
        for file in os.listdir(temp_dir.name):
            file_path = os.path.join(temp_dir.name, file)
            if os.path.isfile(file_path):
                os.unlink(file_path)

        return result

    result = execute_task(prompt)
    return result

def agent_math(prompt):
    result = run_code_task(prompt)
    return result

def process_frame(frame, chalkboard_coords, chalkboard_corners, chalkboard_confirmed, start_time, last_detection_time, average_color, writing_boxes, last_writing_check, last_area, ignored_oranges, chalkboard_roi):
    global orange_count
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

            # Calculate the area of the current and previous bounding boxes
            current_area = gw * gh
            if last_area is None or abs(current_area - last_area) / last_area >= AREA_CHANGE_THRESHOLD:
                # Extract the writing ROI and send to OpenAI for LaTeX conversion
                writing_roi = frame[y+gy:y+gy+gh, x+gx:x+gx+gw]
                latex_text = analyze_image_with_question(writing_roi, "Please write what is written on this chalkboard in latex, just return the full text in latex within a JSON object of the form {'latex': 'your latex text'}")
                print(f"Latex Text detected: {latex_text}")
            last_area = current_area

        # Orange circle detection
        if chalkboard_roi is not None:
            orange_x, orange_y, orange_radius = detect_orange_circle(chalkboard_roi)
            if orange_x is not None and orange_y is not None:
                if all(np.linalg.norm(np.array([orange_x, orange_y]) - np.array([ox, oy])) > ORANGE_THRESHOLD for ox, oy in ignored_oranges):
                    cv2.circle(frame[y:y+h, x:x+w], (orange_x, orange_y), orange_radius, (0, 165, 255), 2)
                    cv2.putText(frame, 'Orange', (orange_x - 10, orange_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

                    # Extract the region close to the orange circle
                    orange_roi = frame[y+orange_y-orange_radius-MARGIN:y+orange_y+orange_radius+MARGIN, x+orange_x-orange_radius-MARGIN:x+orange_x+orange_radius+MARGIN]
                    latex_text = analyze_image_with_question(orange_roi, "Please write what is written on this chalkboard in latex, just return the full text in latex within a JSON object of the form {'latex': 'your latex text'}")
                    print(f"Orange Latex Text detected: {latex_text}")

                    ignored_oranges.append((orange_x, orange_y))

                    if DEBUG_VIEW:
                        cv2.circle(frame[y:y+h, x:x+w], (orange_x, orange_y), ORANGE_THRESHOLD, (0, 0, 255), 2)

                    # Pass the detected LaTeX text to agent_math and handle the result
                    if orange_count > 0:
                        math_result = agent_math(latex_text)
                        if isinstance(math_result, str):
                            print(f"Math Result: {math_result}")
                        elif os.path.isfile(math_result) and math_result.endswith('.png'):
                            print("Displaying generated plot...")
                            plot_image = cv2.imread(math_result)
                            cv2.imshow('Generated Plot', plot_image)
                    else:
                        orange_count += 1
                        print("Orange circle detected, ignoring")

        if PERSON_DETECTION:
            person_box = detect_person(frame)
            if person_box:
                px, py, pw, ph = person_box
                cv2.rectangle(frame, (px, py), (px + pw, py + ph), (0, 0, 255), 2)
                cv2.putText(frame, 'Person', (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return chalkboard_coords, chalkboard_corners, chalkboard_confirmed, start_time, last_detection_time, average_color, writing_boxes, last_writing_check, last_area, ignored_oranges

def main():
    # Initialize the webcam or video feed
    if DEBUG_FEED:
        cap = cv2.VideoCapture(VIDEO_PATH)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    chalkboard_coords = None
    chalkboard_corners = None
    chalkboard_confirmed = False
    start_time = None
    last_detection_time = None
    average_color = None
    writing_boxes = []
    last_writing_check = 0
    last_area = None
    ignored_oranges = []

    while True:
        ret, frame = cap.read()
        if not ret:
            if DEBUG_FEED:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop the video
                continue
            else:
                print("Failed to grab frame.")
                break

        # Only set chalkboard_roi if chalkboard is confirmed
        if chalkboard_confirmed:
            chalkboard_roi = frame[chalkboard_coords[1]:chalkboard_coords[1] + chalkboard_coords[3],
                                   chalkboard_coords[0]:chalkboard_coords[0] + chalkboard_coords[2]]
        else:
            chalkboard_roi = None

        chalkboard_coords, chalkboard_corners, chalkboard_confirmed, start_time, last_detection_time, average_color, writing_boxes, last_writing_check, last_area, ignored_oranges = process_frame(
            frame, chalkboard_coords, chalkboard_corners, chalkboard_confirmed, start_time, last_detection_time, average_color, writing_boxes, last_writing_check, last_area, ignored_oranges, chalkboard_roi)

        # Display the resulting frame
        if DEBUG_VIEW:
            cv2.imshow('Chalkboard and Writing Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
