import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model('model.tf')
#model.summary()

capture = cv2.VideoCapture(0)

if not capture.isOpened():
    raise IOError('Error with the video capture')

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = capture.read()
    frame = cv2.flip(frame, 1)
    image = np.array(frame)
    X = cv2.resize(image, (64,64))
    X = X.reshape((1,64, 64, 3))

    result = model.predict(X, verbose=0)
    

    if result[0][0] == 1:
        cv2.putText(frame, '0', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 0
    elif result[0][1] == 1:
        cv2.putText(frame, '1', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 1
    elif result[0][2] == 1:
        cv2.putText(frame, '2', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 2
    elif result[0][3]:
        cv2.putText(frame, '3', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 3
    elif result[0][4] == 1:
        cv2.putText(frame, '4', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 4
    elif result[0][5] == 1:
        cv2.putText(font, '5', (10,450), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Hand Digit Detection", frame)
        printed = 5
    else:
        cv2.imshow("Hand Digit Detection", frame)
 

    
    


    key = cv2.waitKey(1)
    #27 for Escape
    if key == 27:
        break