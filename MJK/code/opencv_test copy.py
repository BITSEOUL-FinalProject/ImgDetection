import sys
import cv2

face_cascade = cv2.CascadeClassifier("C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_eye.xml")

import numpy as np             
from scipy.misc import imread, imsave

img = imread('./MJK/data/photo/15.jpg')       # 사진을 width x height x rgb 3차원 배열로 나타냄

a = img.sum(axis=2) / 3         # sum(axis=2)는 z축의 값을 다 더한다. (RGB)
imsave('aaa.jpg', a) 

# src = cv2.imread('./MJK/data/photo/src_gray.jpg', cv2.IMREAD_GRAYSCALE)
# src1 = cv2.imread('./MJK/data/photo/src_gray.jpg')
# src_gray = cv2.cvtColor(src1, cv2.COLOR_BGR2GRAY)
# cv2.imwrite('src.jpg', src)

print(src.shape)
# print(src_gray.shape)
# if src is None :
#     print('image load fained')
#     sys.exit()

# src = cv2.resize(src, (0,0), fx=1, fy=1)
# src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# print(faces)



# for (x,y,w,h) in faces:
#     cv2.rectangle(src_gray, (x,y), (x+w, y+h), (255, 0, 0), 2)
#     roi_gray = gray[y:y+h, x:x+w]
#     roi_color = src_gray[y:y+h, x:x+w]
#     eyes = eye_cascade.detectMultiScale(roi_gray)

#     for (ex, ey, ew, eh) in eyes:
#         cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0, 255, 0), 2)

# cv2.imshow('src_gray', src_gray)
# cv2.waitKey()
# cv2.destroyAllWindows()

