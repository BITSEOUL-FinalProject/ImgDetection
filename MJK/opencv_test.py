import cv2
vidcap = cv2.VideoCapture('./Common_data/avi/startup_13.mp4')
success, image = vidcap.read()

# count = 1


# while (vidcap.isOpened()):
#   success, image = vidcap.read()
# #   cv2.imwrite("./Common_data/photo/%d.jpg" %count, image)
# #   print("saved image %d.jpg" % count)


#   print(image)
#   if cv2.waitKey(10) == 27:                    
#       break
#   count += 1
while (vidcap.isOpened()):
  ret, frame = vidcap.read()

  if ret:
        cv2.imshow('video', frame)
        if cv2.waitKey(10) == 27:                    
            break
  else:
      break

vidcap.release()
cv2.destroyAllWindows()