import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

#variables
width, height = 1280, 720
hs, ws = int(120*1.2), int(213*1.2)
folderPath = "Presentation"
imgNumber = 0
last = 7
gestureThreshold = 300
buttonPressed = False
buttonCounter = 0
buttonDelay = 10
annotations = [[]]
annotationNumber = -1
annotationStart = False
max_zoom_level = 5.0
zoom_level = 1.0


#camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#get List of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
#print(pathImages)

#handDetector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    hands, img = detector.findHands(img)
    cv2.line(img, (0, gestureThreshold),(width, gestureThreshold), (0, 255, 0), 1)

    if hands and buttonPressed is False:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand['center']
        lmList = hand['lmList']

        # Constrain values for easier drawing
        indexFinger = lmList[8][0], lmList[8][1]
        xVal = int(np.interp(lmList[8][0], [width//2, width], [0, width]))
        yVal = int(np.interp(lmList[8][1], [150, height-150], [0, height]))
        indexFinger = xVal, yVal

        # if hand is at height of the face
        if cy <= gestureThreshold:
            annotationStart = False
            # Gesture 1 - left
            if fingers == [1, 0, 0, 0, 0]:
                print("left")
                if imgNumber > 0:
                    buttonPressed = True
                    annotations = [[]]
                    annotationNumber = -1
                    imgNumber -= 1

            # Gesture 2 - right
            if fingers == [0, 1, 1, 1, 1]:
                print("right")
                if imgNumber < len(pathImages)-1:
                    buttonPressed = True
                    imgNumber += 1
                    annotations = [[]]
                    annotationNumber = -1

        # Gesture 3 - show pointer
        if fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotationStart = False

        # Gesture 4 - drawing
        if fingers == [0, 1, 0, 0, 0]:
            if annotationStart is False:
                annotationStart = True
                annotationNumber += 1
                annotations.append([])
            cv2.circle(imgCurrent, indexFinger, 12, (0, 0, 255), cv2.FILLED)
            annotations[annotationNumber].append(indexFinger)
        else:
            annotationStart = False

        # Gesture 5 - erase
        if fingers == [1, 1, 1, 1, 1]:
            if annotations:
                annotations.pop(-1)
                annotationNumber -= 1
                buttonPressed = True

        # Gesture 6 - Zoom in (index finger and thumb)
        if fingers == [1, 1, 0, 0, 0]:
            print("Zoom in")
            # Increase the zoom level by 10%
            zoom_level += 0.1
            # Clamp the zoom level to the maximum value
            zoom_level = min(zoom_level, max_zoom_level)
            # Resize the displayed image based on the updated zoom level
            height, width = imgCurrent.shape[:2]
            new_width = int(width * zoom_level)
            new_height = int(height * zoom_level)
            imgCurrent = cv2.resize(imgCurrent, (new_width, new_height))
            print("Zoom level:", zoom_level)

        # Gesture 7 - Zoom out (all fingers closed)
        if fingers == [0, 0, 0, 0, 0]:
            print("Zoom out")
            # Decrease the zoom level by 10%
            zoom_level -= 0.1
            # Clamp the zoom level to the minimum value
            zoom_level = max(zoom_level, 1.0)  # minimum zoom level should be 1.0 to avoid zooming out too much
            # Resize the displayed image based on the updated zoom level
            height, width = imgCurrent.shape[:2]
            new_width = int(width * zoom_level)
            new_height = int(height * zoom_level)
            imgCurrent = cv2.resize(imgCurrent, (new_width, new_height))
            print("Zoom level:", zoom_level)

    else:
        annotationStart = False

    #Button pressed iterations
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter > buttonDelay:
            buttonCounter = 0
            buttonPressed = False

    for i in range(len(annotations)):
        for j in range(len(annotations[i])):
            if j != 0:
                cv2.line(imgCurrent, annotations[i][j-1], annotations[i][j], (0, 0, 200), 12)

    #adding webcam to slides
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w-ws:w] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
