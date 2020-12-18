# 종합_2 load_model
import cv2
import numpy as np

xml_path1 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
xml_path2 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_profileface.xml"
xml_path3 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt.xml"
xml_path4 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt_tree.xml"
xml_path5 = "C:\Anaconda3\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml"

face_classifier  = cv2.CascadeClassifier(xml_path1)
face_classifier2 = cv2.CascadeClassifier(xml_path2)
face_classifier3 = cv2.CascadeClassifier(xml_path3)
face_classifier4 = cv2.CascadeClassifier(xml_path4)
face_classifier5 = cv2.CascadeClassifier(xml_path5)


from tensorflow.keras.models import load_model
model =  load_model("./model/CV05_2_2_MCP_1212-09-0.1764.hdf5")

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    if faces is():
        faces = face_classifier2.detectMultiScale(gray,1.3,5)
        if faces is():
            faces = face_classifier3.detectMultiScale(gray,1.3,5)
            if faces is():
                faces = face_classifier4.detectMultiScale(gray,1.3,5)
                if faces is():
                    faces = face_classifier5.detectMultiScale(gray,1.3,5)
                    if faces is():
                        return img,[] 
                      
    rr = [] 
    xx = []   
    yy = []   

    if len(faces) > 1:
        for j in range(len(faces)):
            for(x,y,w,h) in [faces[j]]:
                cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
                roi = img[y:y+h, x:x+w]
                roi = cv2.resize(roi, (200,200))
                rr.append(roi)
                xx.append(x)
                yy.append(y)                    
        return img,rr,xx,yy 
    else:
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            rr.append(roi)
            return img,rr,x,y   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

    

# ==================================================================================

#파일 경로
FilePath = './teamProject/video/startup.mp4'

#Open the File
movie = cv2.VideoCapture(FilePath) #동영상 핸들 얻기

#Check that the file is opened
if movie.isOpened() == False: #동영상 핸들 확인
    print('Can\'t open the File' + (FilePath))
    exit()



while True:
    #카메라로 부터 사진 한장 읽기 
    ret, frame = movie.read()

    # 얼굴 검출 시도 
    try:
        image, face, x, y = face_detector(frame)
    except:
        image, face       = face_detector(frame)
    Training_Data = []
    try:
        # print(type(face))
        # print(len(face))
        if  len(face) > 1:
            for i in range(len(face)): 
                face[i] = cv2.cvtColor(face[i], cv2.COLOR_BGR2GRAY)
            face = np.array(face)
        else : 
            face[0] = cv2.cvtColor(face[0], cv2.COLOR_BGR2GRAY)
            face = np.array(face)
        
        face = face.reshape(face.shape[0],200,200,1)


        #학습한 모델로 예측시도
        result = model.predict(face)
        
        if len(result) > 1:
            bss1 = []
            for i in range(len(result)):
                if result[i] <= 0.1:
                    bss1.append([cv2.putText(image, "B", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

                elif result[i] >= 0.9 :
                    bss1.append([cv2.putText(image, "N", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)])

                else:
                    bss1.append([cv2.putText(image, "U", (x[i], y[i]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)])
 
            bss1[:]
            cv2.imshow('Face Cropper', image)        
        else:
            if result <= 0.1:
                cv2.putText(image, "B", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
            elif result >= 0.9:
                cv2.putText(image, "N", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Face Cropper', image)
            else:
                cv2.putText(image, "U", (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow('Face Cropper', image)
    except:
        #얼굴 검출 안됨 
        cv2.putText(image, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(2)==13:
        break
movie.release()
cv2.destroyAllWindows()



