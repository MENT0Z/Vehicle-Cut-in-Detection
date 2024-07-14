from ultralytics import YOLO
import cv2
import cvzone
import time
import math
import numpy as np
from sort import *

vdo = "videos/test.mp4"
cap = cv2.VideoCapture(vdo)
cap.set(3, 1280)
cap.set(4, 720)
model = YOLO('yolov8n.pt')

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

# Assuming your car's speed is 38 km/h we get it from the sensors attacted to our car
v1 = 38  # in km/h

# Convert speed to m/s
v1_mps = v1 * (1000 / 3600)  # km/h to m/s

# Dictionary to store historical positions, times, and speeds for each tracked vehicle
vehicle_data = {}

# Define the fixed position for your car (origin in the coordinate system)
car_position = (0, 0)

# Tracking
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Parameters for speed averaging
speed_buffer_size = 5  # Number of recent speed measurements to average

while True:
    newFrameTime = time.time()
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = np.empty((0, 5))
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w // 2, y1 + h // 2
            cvzone.cornerRect(img, (x1, y1, w, h))
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]
            if currentClass in ["car", "truck", "bus", "motorbike"] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)

    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2

        # Update vehicle data
        if id not in vehicle_data:
            vehicle_data[id] = {'positions': [], 'speeds': []}
        vehicle_data[id]['positions'].append((newFrameTime, cx, cy))

        # Calculate velocity of the vehicle
        if len(vehicle_data[id]['positions']) > 1:
            t1, x1, y1 = vehicle_data[id]['positions'][-2]
            t2, x2, y2 = vehicle_data[id]['positions'][-1]
            dt = t2 - t1
            dx = x2 - x1
            dy = y2 - y1
            distance_pixels = math.sqrt(dx**2 + dy**2)
            # Convert pixel distance to real distance (you need to calibrate this value)
            real_distance_meters = distance_pixels*0.7 # example calibration factor
            velocity_mps = real_distance_meters / dt
            velocity_kph = velocity_mps * 3.6

            # Add the new speed to the buffer
            vehicle_data[id]['speeds'].append(velocity_kph)
            # Maintain the buffer size
            if len(vehicle_data[id]['speeds']) > speed_buffer_size:
                vehicle_data[id]['speeds'].pop(0)
            # Calculate the average speed
            avg_speed_kph = sum(vehicle_data[id]['speeds']) / len(vehicle_data[id]['speeds'])
            print(f"Vehicle {id} average speed: {avg_speed_kph:.2f} km/h")
            cvzone.putTextRect(img, f'vel: {avg_speed_kph:.2f}km/h', (cx, cy - 50),
                               scale=2, thickness=2, offset=10, colorR=(0, 0, 255))

            # Calculate time to collision
            if v1_mps > velocity_mps:
                relative_velocity_mps = v1_mps - velocity_mps
                distance_to_vehicle_meters = real_distance_meters
                time_to_collision = (distance_to_vehicle_meters / relative_velocity_mps)*10
                print(f"Time to collision with vehicle {id}: {time_to_collision:.2f} seconds")
                cvzone.putTextRect(img, f'TTC: {time_to_collision:.2f}s', (cx, cy - 100),
                                   scale=2, thickness=2, offset=10, colorR=(255, 0, 0))

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
