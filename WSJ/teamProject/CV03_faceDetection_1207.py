import cv2
import numpy as np
import matplotlib.pyplot as plt
# import os
# print(os.getcwd())

xml_path1 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml"
xml_path2 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_profileface.xml"


image_path1 = "./teamProject/images/test4.jpg"

# Load the cascade
face_cascade = cv2.CascadeClassifier(xml_path2)
# 얼굴인식에 필요한 사진들을 수치화해서 xml로 만든듯


# Read the input image
img = cv2.imread(image_path1)
# cv2로 이미지 불러오기

# plt.figure(figsize=(12,10))
# plt.imshow(img,cmap="gray")
# plt.show()

# Detect faces
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1,minNeighbors=4)
# detectMultiScale : 
# scaleFactor : 
# minNeighbors :이미지 피라미드에 의한 여러 스케일의 크기에서 
#               minNeighbors 횟수 이상 검출된 object는 valid하게 검출할 때 쓰인다


# Draw rectangle around the faces
for (x, y, w, h) in faces: 
  cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

# (255,0,0) 직사각형 색
# (2) 선의 두께

# cv2.imshow("img",img)

# 결과 내보내기
# cv2.imwrite ("./teamProject/images/test3_d.png", img) 
# print ( '성공적으로 저장 됨')
plt.figure(figsize=(12,10))
plt.imshow(img)
plt.show()


