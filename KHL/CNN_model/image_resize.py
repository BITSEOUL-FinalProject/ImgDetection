import glob
from cv2 import cv2
import numpy as np

# 수지
path1 = glob.glob("D:/Suzy/sort/*.jpg")
cv_img1 = []
index1 = 0
for img in path1:
    read_image1 = cv2.imread(img)
    resize_image1 = cv2.resize(read_image1, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    bgr_to_rgb1 = cv2.cvtColor(resize_image1, cv2.COLOR_BGR2RGB)
    cv2.imwrite('D:/Suzy/sort/'+ str(index1) +'.jpg' , resize_image1)
    cv_img1.append(resize_image1)
    index1 += 1
print("train 불러오기, resize 완료!")

train_images = np.array(cv_img1)
print(train_images.shape)           

# 남주혁
path2 = glob.glob("D:/Nam/sort/*.jpg")
cv_img2 = []
index2 = 0
for img in path2:
    read_image2 = cv2.imread(img)
    resize_image2 = cv2.resize(read_image2, dsize=(224, 224), interpolation=cv2.INTER_AREA)
    bgr_to_rgb2 = cv2.cvtColor(resize_image2, cv2.COLOR_BGR2RGB)
    cv2.imwrite('D:/Nam/sort/'+ str(index2) +'.jpg', resize_image2)
    cv_img2.append(resize_image2)
    index2 += 1
print("test 불러오기, resize 완료!")

test_images = np.array(cv_img2)
print(test_images.shape)           