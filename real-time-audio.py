# https://www.pysimplegui.org/en/latest/#jump-start
from PySimpleGUI import Window, Button, Text, Image

import numpy as np
import time
import cv2
import subprocess
from gtts import gTTS
from pydub import AudioSegment
layout = [[Text(text='Blind Assistance System',
                font=('Arial Bold', 16),
                size=20, expand_x=True,
                justification='center')],
            [Image('blind.png',
                   expand_x=True, expand_y=True)],
            [Button('Run')],
            [Text(text_color='black', key='out', justification='center')]
            ]


# Create the Window
window = Window('VTU', layout)
# Event Loop to process "events"
while True:
    event, values = window.read()
    temp=event
    # When press Run
    if event == 'Run' :
        LABELS = open("coco.names").read().strip().split("\n")

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # initialize
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        frame_count = 0
        start = time.time()
        first = True
        frames = []

        while True:
            frame_count += 1
            # Capture frame-by-frameq
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            frames.append(frame)

            # if frame_count == 300:
            #	break
            if ret:
                key = cv2.waitKey(1)
                if frame_count % 60 == 0:
                    end = time.time()
                    # grab the frame dimensions and convert it to a blob
                    (H, W) = frame.shape[:2]
                    # construct a blob from the input image and then perform a forward
                    # pass of the YOLO object detector, giving us our bounding boxes and
                    # associated probabilities
                    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                                 swapRB=True, crop=False)
                    net.setInput(blob)
                    layerOutputs = net.forward(ln)

                    # initialize our lists of detected bounding boxes, confidences, and
                    # class IDs, respectively
                    boxes = []
                    confidences = []
                    classIDs = []
                    centers = []

                    # loop over each of the layer outputs
                    for output in layerOutputs:
                        # loop over each of the detections
                        for detection in output:
                            # extract the class ID and confidence (i.e., probability) of
                            # the current object detection
                            scores = detection[5:]
                            classID = np.argmax(scores)
                            confidence = scores[classID]

                            # filter out weak predictions by ensuring the detected
                            # probability is greater than the minimum probability
                            if confidence > 0.5:
                                # scale the bounding box coordinates back relative to the
                                # size of the image, keeping in mind that YOLO actually
                                # returns the center (x, y)-coordinates of the bounding
                                # box followed by the boxes' width and height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top and
                                # and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates, confidences,
                                # and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)
                                centers.append((centerX, centerY))

                    # apply non-maxima suppression to suppress weak, overlapping bounding
                    # boxes
                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

                    texts = []
                    cv2.imshow('Image', frame)
                    cv2.waitKey(1)
                    # ensure at least one detection exists
                    if len(idxs) > 0:
                        # loop over the indexes we are keeping
                        for i in idxs.flatten():
                            # find positions
                            centerX, centerY = centers[i][0], centers[i][1]

                            if centerX <= W / 3:
                                W_pos = "left "
                            elif centerX <= (W / 3 * 4):
                                W_pos = "center "
                            else:
                                W_pos = "right "

                            if centerY <= H / 3:
                                H_pos = "top "
                            elif centerY <= (H / 3 * 4):
                                H_pos = "mid "
                            else:
                                H_pos = "bottom "
                            texts.append(H_pos + W_pos + LABELS[classIDs[i]])

                    print(texts)

                    if texts:
                        description = ', '.join(texts)
                        tts = gTTS(description, lang='en')
                        tts.save("tts.mp3")
                        tts = AudioSegment.from_mp3("C:/Users/Deepa/Desktop/yolo-master/tts.mp3")
                        subprocess.call(["ffplay", "-nodisp", "-autoexit", "C:/Users/Deepa/Desktop/yolo-master/tts.mp3"])
        cap.release()
        cv2.destroyAllWindows()
        window['out'].update(str(values))
# Close window
window.close()