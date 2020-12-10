import numpy as np
from cv2 import cv2 # from cv2 import cv2 나는 이렇게 적어야 실행된다
import matplotlib.pyplot as plt 
# %matplotlib inline

image = cv2.imread('./img/start.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

xml1 = './haarcascades_cuda/haarcascade_profileface.xml'
xml2 = './haarcascades_cuda/haarcascade_frontalface_default.xml'

# 수지 얼굴 프로필 페이스
face_cascade = cv2.CascadeClassifier(xml1)
faces = face_cascade.detectMultiScale(gray, 1.2, 5)

print("Number of faces detected: " + str(len(faces)))

if len(faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)


# 남주혁 얼굴 디폴트
face_cascade = cv2.CascadeClassifier(xml2)
faces = face_cascade.detectMultiScale(gray, 1.2, 5)

print("Number of faces detected: " + str(len(faces)))

if len(faces):
    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), cmap='gray')
plt.xticks([]), plt.yticks([]) 
plt.show()

