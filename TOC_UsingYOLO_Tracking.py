from ultralytics import YOLO
import cv2
from collections import defaultdict
import numpy as np

# List of classes to track
track_classes = ["car", "truck", "bus", "motorbike"]
# Model file path
model = YOLO("yolov8n.pt")
# Video file path
cap = cv2.VideoCapture('videos/test.mp4')
cap.set(3, 1280)
cap.set(4, 720)
track_history = defaultdict(lambda: [])
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

# Conversion factor from pixels per frame to km/h
pixels_to_kmh = 0.036  # This is an arbitrary scale factor; adjust based on actual measurement
# Conversion factor from pixels to meters
pixels_to_meters = 0.05  # This is an arbitrary scale factor; adjust based on actual measurement

# List of class indices to track (for COCO dataset)
class_names = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
               "teddy bear", "hair drier", "toothbrush"]

class_list = [class_names.index(cls) for cls in track_classes]

# Velocity of the car with the camera (in km/h)
camera_car_velocity = 40  # km/h

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model.track(frame, show=False, verbose=False, conf=0.4, classes=class_list, persist=True)
        
        if results and results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            annotated_frame = results[0].plot()

            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((x, y))
                if len(track) > 1:
                    # Calculate velocity in pixels per second
                    dx = track[-1][0] - track[-2][0]
                    dy = track[-1][1] - track[-2][1]
                    distance_px = np.sqrt(dx**2 + dy**2)
                    vehicle_velocity_pxps = distance_px * fps  # pixels per second

                    # Convert velocity to km/h
                    vehicle_velocity_kmh = vehicle_velocity_pxps * pixels_to_kmh

                    # Calculate relative velocity (vehicle's velocity minus camera car's velocity)
                    relative_velocity_kmh =   camera_car_velocity-vehicle_velocity_kmh

                    # Calculate distance to the vehicle in meters
                    distance_to_vehicle_px = y  # assuming y-coordinate represents the distance to the vehicle
                    distance_to_vehicle_m = distance_to_vehicle_px * pixels_to_meters

                    # Calculate time to collision (TTC)
                    relative_velocity_mps = relative_velocity_kmh / 3.6  # convert km/h to m/s
                    ttc = distance_to_vehicle_m / relative_velocity_mps if relative_velocity_mps != 0 else float('inf')
                    ttc = ttc/2.5
                    # Display velocity and TTC on the frame
                    # cv2.putText(annotated_frame, f'ID {track_id}: {vehicle_velocity_kmh:.2f} km/h', (int(x), int(y - 30)),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f'TTC: {ttc:.2f} s', (int(x), int(y - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Warning if TTC is less than threshold seconds
                    if ttc < 0.55:
                        cv2.putText(annotated_frame, 'WARNING: Collision in < 0.6s!', (int(x), int(y + 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
