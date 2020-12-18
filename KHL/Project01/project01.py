from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from cv2 import cv2
import os
import cvlib as cv
                    
# load model
model = load_model('D:/ImgDetection/KHL/Checkpoint/cp-inc-rmsprop-58-0.599426.hdf5')

# open video
video = cv2.VideoCapture("D:/Sample.mp4")

classes = ['suzy','juhyuk','sunho','hanna']

# loop through frames
while video.isOpened():

    # read frame from video 
    status, frame = video.read()

    # apply face detection
    face, confidence = cv.detect_face(frame)
    faces = np.array(face)

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
    cv2.imshow("face recognition", frame)

    # press "Q" to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release resources
video.release()
cv2.destroyAllWindows() 

#     # loop through detected faces
#     for _, f in enumerate(face):

#         # get corner points of face rectangle        
#         (startX, startY) = f[0], f[1]
#         (endX, endY) = f[2], f[3]
        
#         # draw rectangle over face
#         cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)

#         # crop the detected face region
#         face_crop = np.copy(frame[startY:endY,startX:endX])

#         if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
#             continue

#         # preprocessing for face detection model
#         face_crop = cv2.resize(face_crop, (200,200))
#         face_crop = face_crop.astype("float") / 255.0
#         face_crop = np.expand_dims(face_crop, axis=0)

#         # apply face detection on face
#         conf = model.predict(face_crop)[0] 

#         if np.max(conf) == conf[0]:
#             label = "{}: {:.2f}%".format(classes[0], conf[0] * 100)
#         elif np.max(conf) == conf[1]:
#             label = "{}: {:.2f}%".format(classes[1], conf[1] * 100)
#         elif np.max(conf) == conf[2]:
#             label = "{}: {:.2f}%".format(classes[2], conf[2] * 100)
#         else:    
#             label ="{}: {:.2f}%".format(classes[3], conf[3] * 100)

#         # write label and confidence above face rectangle
#         cv2.putText(frame, label, (startX, startY-10),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # display output
#     cv2.imshow("face recognition", frame)

#     # press "Q" to stop
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # release resources
# video.release()
# cv2.destroyAllWindows()

