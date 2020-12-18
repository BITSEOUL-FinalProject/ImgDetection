from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from cv2 import cv2
import os
import cvlib as cv
                    
# load model
model = load_model('D:/ImgDetection/KHL/Checkpoint/rmsprop_best.hdf5')

# open video
video = cv2.VideoCapture("D:/Sample6.mp4")
# video2 = cv2.VideoCapture("D:/Sample6.mp4")

classes = ['suzy','juhyuk','sunho','hanna']

# loop through frames
# while (video.isOpened() or video2.isOpened()):
while video.isOpened():
    # read frame from video 
    status, frame = video.read()
    # status2, frame2 = video2.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)
    faces = np.array(face)

    # face2, confidence2 = cv.detect_face(frame2)
    # faces2 = np.array(face2)

    # total = zip(faces, faces2)

    # for face, face2 in total:
    #     # get corner points of face rectangle        
    #     (startX, startY) = face[0], face[1]
    #     (endX, endY) = face[2], face[3]

    #     (startX_2, startY_2) = face2[0], face2[1]
    #     (endX_2, endY_2) = face2[2], face2[3]
            
    #     # draw rectangle over face
    #     cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
    #     cv2.rectangle(frame2, (startX_2,startY_2), (endX_2,endY_2), (0,255,0), 2)

    #     # crop the detected face region
    #     face_crop = np.copy(frame[startY:endY,startX:endX])
    #     face_crop2 = np.copy(frame2[startY:endY,startX:endX])

    #     if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
    #         continue

    #     # preprocessing for face detection model
    #     face_crop = cv2.resize(face_crop, (200,200))
    #     face_crop = face_crop.astype("float") / 255.0
    #     face_crop = np.expand_dims(face_crop, axis=0)

    #     face_crop2 = cv2.resize(face_crop2, (200,200))
    #     face_crop2 = face_crop2.astype("float") / 255.0
    #     face_crop2 = np.expand_dims(face_crop2, axis=0)

    #     # apply face detection on face
    #     conf = model.predict(face_crop)[0] 
    #     conf2 = model.predict(face_crop2)[0] 
        
    #     if np.max(conf) == conf[0]:
    #         label = "{}: {:.2f}%".format(classes[0], conf[0] * 100)
    #     elif np.max(conf) == conf[1]:
    #         label = "{}: {:.2f}%".format(classes[1], conf[1] * 100)
    #     elif np.max(conf) == conf[2]:
    #         label = "{}: {:.2f}%".format(classes[2], conf[2] * 100)
    #     else:    
    #         label ="{}: {:.2f}%".format(classes[3], conf[3] * 100)

    #     if np.max(conf2) == conf2[0]:
    #         label2 = "{}: {:.2f}%".format(classes[0], conf2[0] * 100)
    #     elif np.max(conf2) == conf2[1]:
    #         label2 = "{}: {:.2f}%".format(classes[1], conf2[1] * 100)
    #     elif np.max(conf2) == conf2[2]:
    #         label2 = "{}: {:.2f}%".format(classes[2], conf2[2] * 100)
    #     else:    
    #         label2 ="{}: {:.2f}%".format(classes[3], conf2[3] * 100)

    #     # # write label and confidence above face rectangle
    #     cv2.putText(frame, label, (startX, startY-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #     cv2.putText(frame2, label2, (startX_2, startY_2-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    for face in faces:
        
        # get corner points of face rectangle        
        (startX, startY) = face[0], face[1]
        (endX, endY) = face[2], face[3]
            
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

        if np.max(conf) == conf[0]:
            label = "{}: {:.2f}%".format(classes[0], conf[0] * 100)
        elif np.max(conf) == conf[1]:
            label = "{}: {:.2f}%".format(classes[1], conf[1] * 100)
        elif np.max(conf) == conf[2]:
            label = "{}: {:.2f}%".format(classes[2], conf[2] * 100)
        else:    
            label ="{}: {:.2f}%".format(classes[3], conf[3] * 100)

        # write label and confidence above face rectangle
        cv2.putText(frame, label, (startX, startY-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
    # display output
    # frame = cv2.resize(frame, (720,480))

    cv2.imshow("face recognition", frame)
    # cv2.imshow("face recognition2", frame2)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
video.release()
# video2.release()
cv2.destroyAllWindows() 