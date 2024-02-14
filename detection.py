import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone

# Loading data from a pickle file named "Maruf"
with open("Maruf", "rb") as f:
    data = pickle.load(f)
    polylines, polygone_num = data['polylines'], data['polygone_num']

# Reading class labels from a text file named "coco.txt"
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Loading YOLO model
model = YOLO('yolov8s.pt')

# Opening a video capture device
cap = cv2.VideoCapture()

count = 0

# Running a loop to process each frame of the video
while True:
    ret, frame = cap.read()  # Reading a frame from the video
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restarting video if it ends
        continue

    count += 1
    if count % 3 != 0:
        continue  # Skipping every 3rd frame

    frame = cv2.resize(frame, (1020, 500))  # Resizing the frame

    # Making a copy of the frame
    frame_copy = frame.copy()

    # Performing object detection using YOLO model
    results = model.predict(frame)

    # Extracting bounding box coordinates and class predictions
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    # List to store center points of detected cars
    list1 = []

    # Iterating over bounding box data
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])  # class index

        c = class_list[d]  # getting class label
        cx = int(x1 + x2) // 2  # calculating center x
        cy = int(y1 + y2) // 2  # calculating center y
        if 'car' in c:
            list1.append((cx, cy))  # Adding center point to list if class is 'car'

    counter1 = []  # List to store detected car centers within polygons
    list2 = []  # List to store polygon indexes

    # Iterating over polygons
    for i, polyline in enumerate(polylines):
        list2.append(i)
        cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)  # Drawing polygons
        cvzone.putTextRect(frame, f'{polygone_num[i]}', tuple(polyline[0]), 1, 1)  # Adding polygon number as text
        for i1 in list1:
            cx1 = i1[0]  # object center x
            cy1 = i1[1]  # object center y
            result = cv2.pointPolygonTest(polyline, ((cx1, cy1)), False)  # Checking if object center is within polygon
            if result >= 0:
                cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)  # Drawing circle at object center
                cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)  # Highlighting polygon
                counter1.append(cx1)  # Adding detected car center to the list

    # Counting cars inside polygons and calculating free space
    car_count = len(counter1)
    free_space = len(list2) - car_count

    # Adding text displaying car count and free space on the frame
    cvzone.putTextRect(frame, f'CARS_IN_AREA:-{car_count}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'FREE_SPACE:-{free_space}', (50, 100), 2, 2)

    # Displaying the frame
    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1) & 0xFF

# Releasing video capture device and closing all windows
cap.release()
cv2.destroyAllWindows()


