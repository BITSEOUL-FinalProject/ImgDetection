from cv2 import cv2
import numpy as np

test = cv2.imread("D:/ImgDetection/KHL/2.jpg")
# a = np.array([[[1,2,3], [3,4,5], [5,6,7], [8,9,10]],
#               [[2,3,4], [4,5,6], [7,8,9], [10,11,12]],
#               [[3,4,5], [6,7,8], [9,10,11], [12,13,14]],
#               [[4,5,6], [7,8,9], [10,11,12], [13,14,15]]])
# print(a)

# b, g, r = cv2.split(a)

# print("\n",b)
# b, g, r = cv2.split(test)
# print(b.shape)
# print(test[0][0])
test = cv2.cvtColor(test,  cv2.COLOR_BGR2GRAY)
print(test.shape)
test = cv2.merge((test, test, test))
print(test.shape)
# print(test[0][0])

cv2.imshow('test', test)
cv2.waitKey(0)
cv2.destroyAllWindows()

