from cv2 import cv2
import numpy as np
import glob

xml_path1 = "C:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"
xml_path2 = "C:/Anaconda3/Lib/site-packages/cv2/data/haarcascade_profileface.xml"

face_cascade = cv2.CascadeClassifier(xml_path1)
eye_cascade = cv2.CascadeClassifier(xml_path2)

path = glob.glob("D:/naver/1/*.jpg")
a = []
count = 571
for image in path:
    img = cv2.imread(image)
    try:
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            cropped = img[y - int(h/4):y + h + int(h/4), x - int(w/4):x + w + int(w/4)]
            cv2.imwrite("D:/Nam/naver/"+str(count)+".jpg", cropped)
            # roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_color)
        count += 1
    except:
        pass

path = glob.glob("D:/Suzy/google/*.jpg")
count = 0

for image in path:
    img = cv2.imread(image)
    cv2.imwrite("D:/Suzy/sort/" + str(count) + ".jpg", img)
    count += 1

path1 = glob.glob("D:/Nam/google/*.jpg")
count1 = 0

for image in path1:
    img = cv2.imread(image)
    cv2.imwrite("D:/Nam/sort/" + str(count1) + ".jpg", img)
    count1 += 1
    
# cv2.imshow("image1",image[369])
# cv2.waitKey(0)