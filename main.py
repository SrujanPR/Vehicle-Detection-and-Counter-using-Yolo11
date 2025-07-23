import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
from ultralytics import YOLO
from collections import defaultdict

model = YOLO('yolo11l.pt')
class_list = model.names
cap = cv2.VideoCapture('./test videos/test video_1.mp4')

# Get Video Properties
frame_width = int(cap.get(3))  # Width
frame_height = int(cap.get(4))  # Height
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define VideoWriter to Save the Output Video
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Red Line Position
line_y_red = 430

# Dictionary to Store Object Counts
class_counts = defaultdict(int)

# Set to Keep Track of IDs that Crossed the Line
crossed_ids = set()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO Tracking on the Video Frame
    results = model.track(frame, persist=True, classes=[1,2,3,5,6,7])

    if results[0].boxes is not None and len(results[0].boxes) > 0:
        # Get Bounding Boxes, Class Indices, and Track IDs
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        class_indices = results[0].boxes.cls.int().cpu().tolist()

        # Draw Red Line
        cv2.line(frame, (690, line_y_red), (1130, line_y_red), (0, 0, 255), 3)
        cv2.putText(frame, 'Red Line', (690, line_y_red - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Loop Through Detections
        for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            class_name = class_list[class_idx]

            # Draw Detection Box & ID
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"ID: {track_id} {class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Check if Object Crossed the Red Line
            if cy > line_y_red and track_id not in crossed_ids:
                crossed_ids.add(track_id)
                class_counts[class_name] += 1

    # Display the Object Counts on the Frame
    y_offset = 30
    for class_name, count in class_counts.items():
        cv2.putText(frame, f"{class_name}: {count}", (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        y_offset += 30

    # Write Frame to Output Video
    out.write(frame)
    cv2.imshow("YOLO Object Tracking and Counting", frame)

    # Press 'Q' to Exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Processed video saved as {output_path}")
