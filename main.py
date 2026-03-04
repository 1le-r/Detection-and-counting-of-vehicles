import cv2
import numpy as np
import os

VIDEO_PATH = os.path.join("video", "Video.mp4")

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: Video not found. Please add your video inside the 'video' folder and name it 'Video.mp4'")
    exit()

fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)

counter_line_position = 450
offset = 6
vehicle_count = 0

trackers = []
vehicle_id = 0
counted_ids = set()


def get_center(x, y, w, h):
    return int(x + w / 2), int(y + h / 2)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (800, 600))
    mask = fgbg.apply(frame)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 1500:
            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h))

    new_trackers = []
    for det in detections:
        x, y, w, h = det
        cx, cy = get_center(x, y, w, h)

        matched = False
        for t in trackers:
            tid, tx, ty = t
            if abs(cx - tx) < 50 and abs(cy - ty) < 50:
                new_trackers.append((tid, cx, cy))
                matched = True
                break

        if not matched:
            vehicle_id += 1
            new_trackers.append((vehicle_id, cx, cy))

    trackers = new_trackers

    for tid, cx, cy in trackers:
        if (counter_line_position - offset) < cy < (counter_line_position + offset):
            if tid not in counted_ids:
                vehicle_count += 1
                counted_ids.add(tid)

    cv2.line(frame, (0, counter_line_position), (800, counter_line_position), (255, 0, 0), 2)

    for tid, cx, cy in trackers:
        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
        cv2.putText(frame, str(tid), (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f'Vehicle Count: {vehicle_count}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Vehicle Detection and Counting", frame)

    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
