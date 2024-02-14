import cv2
import numpy as np
import cvzone
import pickle

# Opening a video capture device
cap = cv2.VideoCapture()

drawing = False  # Flag to track if drawing is in progress
polygone_num = []  # List to store polygon numbers
try:
    # Trying to load data from a pickle file named "Maruf"
    with open("Maruf", "rb") as f:
        data = pickle.load(f)
        polylines, polygone_num = data['polylines'], data['polygone_num']
except:
    polylines = []  # Initializing list of polylines if pickle file is not found
points = []  # List to store drawn points
current_name = " "  # Variable to store current polygon number

# Function to handle mouse events for drawing polygons
def draw(event, x, y, flags, param):
    global points, drawing, current_name
    drawing = True
    if event == cv2.EVENT_LBUTTONDOWN:
        points = [(x, y)]  # Storing the starting point of the polygon
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))  # Appending points as mouse moves
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_name = input('number of polygon:-')  # Asking for the number of the polygon
        if current_name:
            polygone_num.append(current_name)  # Adding the entered number to the list
            # Converting drawn points into numpy array and appending to the list of polylines
            polylines.append(np.array(points, np.int32))

while True:
    ret, frame = cap.read()  # Reading a frame from the video
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restarting video if it ends
        continue
    frame = cv2.resize(frame, (1020, 500))  # Resizing the frame

    # Drawing existing polygons on the frame
    for i, polyline in enumerate(polylines):
        cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)  # Drawing polygons
        cvzone.putTextRect(frame, f'{polygone_num[i]}', tuple(polyline[0]), 1, 1)  # Adding polygon number as text

    cv2.imshow('FRAME', frame)  # Displaying the frame
    cv2.setMouseCallback('FRAME', draw)  # Setting mouse event callback function to draw polygons

    Key = cv2.waitKey(100) & 0xFF
    if Key == ord('s'):  # Saving drawn polygons if 's' key is pressed
        with open("Maruf", "wb") as f:
            data = {'polylines': polylines, 'polygone_num': polygone_num}
            pickle.dump(data, f)
cap.release()  # Releasing video capture device
cv2.destroyAllWindows()  # Closing all windows
