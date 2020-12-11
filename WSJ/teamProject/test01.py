from tensorflow.keras.applications import inception_resnet_v2
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization
# images = './WSJ/teamProject/images4/0/test/ad/315.jpg'
# image = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
# print("gray : ",image.shape)     
# cv2.imshow("gray",image)  

# images = './WSJ/teamProject/images4/0/test/ad/315.jpg'
# images = cv2.imread(images)
# print("none : ",images.shape)
# print(images)
# cv2.imshow("none",images)   
 
# 200,200 -> read -> 3개 -> gray 

images = 'D:\\ImgDetection\\WSJ\\teamProject\\images2\\0\\916.jpg'
images = cv2.imread(images,-1)
print("none : ",images.shape)
cv2.imshow("none",images)   
cv2.waitKey()

