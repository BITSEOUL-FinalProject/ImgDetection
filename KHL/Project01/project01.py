from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from cv2 import cv2
import os
import cvlib as cv
                    
# load model
model = load_model('D:/ImgDetection/KHL/Checkpoint/cp_inc-33-0.095160.hdf5')

# open video
video = cv2.VideoCapture("D:/Sample.mp4")

classes = ['juhyuk','suzy','unknown']

# loop through frames
while video.isOpened():

    # read frame from video 
    status, frame = video.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)

    # loop through detected faces
    for idx, f in enumerate(face):

        # get corner points of face rectangle        
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        
        # draw rectangle over face
        cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

        # crop the detected face region
        face_crop = np.copy(frame[startY:endY,startX:endX])

        if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
            continue

        # preprocessing for face detection model
        face_crop = cv2.resize(face_crop, (200,200))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = np.expand_dims(face_crop, axis=0)
        # apply face detection on face
        conf = model.predict(face_crop)[0] 
       
        label = "{}: {:.2f}%".format(classes[0], conf[0] * 100)
        label2 = "{}: {:.2f}%".format(classes[1], 100 - conf[0] * 100)

        # Y = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (endX+5, startY+20),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, label2, (endX+5, startY+50),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # display output
    cv2.imshow("face recognition", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
video.release()
cv2.destroyAllWindows()