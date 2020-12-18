from cv2 import cv2

cap = cv2.VideoCapture("D:/Sample.mp4")
cap1 = cv2.VideoCapture("D:/Sample6.mp4")

if cap.isOpened()== False:
    print("Error camera 1 isn't connecting")
if cap1.isOpened()== False:
    print("Error camera 2 isn't connecting")

while (cap.isOpened() or cap1.isOpened()):
    ret, img = cap.read()
    ret1, img1 = cap1.read()
    img = cv2.resize(img, (720, 480))
    img1 = cv2.resize(img1, (720, 480))
    cv2.rectangle(img, (100,100), (400,400), (0,255,0), 2)
    cv2.rectangle(img1, (100,100), (400,400), (0,255,0), 2)

    if ret == True:
        cv2.imshow('Video 1',img)
        cv2.imshow('Video 2',img1)

    if cv2.waitKey(20) and 0xFF == ord('q'):
            break

cap.release()
cap1.release()
cv2.destroyAllWindows()