from cv2 import cv2
import time
import sys

font = cv2.FONT_HERSHEY_SIMPLEX
capture = cv2.VideoCapture("C:/Users/bitcamp/Desktop/test1.mp4")

cascade_file = "C:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
cascade_file2 = "C:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_profileface.xml"
 
frontFace_cascade = cv2.CascadeClassifier(cascade_file)
profileFace_cascade = cv2.CascadeClassifier(cascade_file2)


while True:
    # if(capture.get(cv2.CAP_PROP_POS_FRAMES) == capture.get(cv2.CAP_PROP_FRAME_COUNT)):
    #     capture.open("Image/Star.mp4")

    ret, frame = capture.read()
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    front_faces = frontFace_cascade.detectMultiScale(grayframe,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in front_faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame,'front_face',(x-5,y-5),font,0.5,(255,0,0),2)
       
    profile_faces = profileFace_cascade.detectMultiScale(grayframe,scaleFactor=1.3,minNeighbors=5)
    for (x,y,w,h) in profile_faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame,'profile_face',(x-5,y-5),font,0.5,(255,0,0),2)
   
    cv2.imshow("VideoFrame", frame)

    if cv2.waitKey(1) > 0: break

capture.release()
cv2.destroyAllWindows()



