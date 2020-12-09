import cv2


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
    for(x,y,w,h) in faces:
        try:  
            print(faces[0])  
            print("111111111111",faces[1])  
            if faces[1] is not None:
                print("ifif")
                for j in range(len(faces)):
                    print("forfor")
                    print(j)
                    for(x,y,w,h) in [faces[j]]:
                        print("x,y,w,h",x,y,w,h)
                        print("forforforfor")
                        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
                        roi = img[y:y+h, x:x+w]
                        roi = cv2.resize(roi, (200,200))
                        rr = rr.append(roi)
                        print("rrrr",rr)
                return img,rr    
        except:
            cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,255),2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))
            return img,roi   #검출된 좌표에 사각 박스 그리고(img), 검출된 부위를 잘라(roi) 전달

    

# ==================================================================================

#Open the File
vidcap = cv2.VideoCapture('./Common_data/avi/startup_13.mp4') 

#Check that the file is opened
if vidcap.isOpened() == False: #동영상 핸들 확인
    print('Can\'t open the File' + (FilePath))
    exit()

#create the window & change the window size
#윈도우 생성 및 사이즈 변경
# cv2.namedWindow('Face')


while True:
    #카메라로 부터 사진 한장 읽기 
    ret, frame = vidcap.read()

    # 얼굴 검출 시도 
    image, face = face_detector(frame)
    try:
        if face[1] is not None:
            face = np.array(face)
    except:
        pass
    Training_Data = []
    try:
        #검출된 사진을 흑백으로 변환 
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        Training_Data.append(np.asarray(face, dtype=np.uint8))
        face = np.array(Training_Data)
        face = face.reshape(face.shape[0],100,100,4)

        #위에서 학습한 모델로 예측시도
        result = model.predict(face)
        # result = np.argmax(result)
        #result[1]은 신뢰도이고 0에 가까울수록 자신과 같다는 뜻이다. 
        # if result[1] < 500:
        #     #????? 어쨋든 0~100표시하려고 한듯 
        #     confidence = int(100*(1-(result[1])/300))
        #     # 유사도 화면에 표시 
        #     display_string = str(confidence)+'% Confidence it is user'

        # cv2.putText(image,display_string,(100,120), cv2.FONT_HERSHEY_COMPLEX,1,(250,120,255),2)
        #75 보다 크면 동일 인물로 간주해 UnLocked! 
        try: 
            if result[1] is not None:
                print("if")
                for i in range(len(result)):
                    print("for")
                    if result[i] >= 0.0 and result[i] < 0.02:
                        print(result)
                        cv2.putText(image, "B", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face Cropper', image)
                    elif result[i] >= 0.98 and result[i] <= 1.02:
                        print(result)
                        cv2.putText(image, "N", (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow('Face Cropper', image)
                    else:
                        print(result)
                        #75 이하면 타인.. Locked!!! 
                        cv2.putText(image, "U", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                        cv2.imshow('Face Cropper', image)
        except:
            pass
        if result >= 0.0 and result < 0.02:
            print(result)
            cv2.putText(image, "B", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        elif result >= 0.98 and result <= 1.02:
            print(result)
            cv2.putText(image, "N", (300, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Face Cropper', image)
        else:
            print(result)
            #75 이하면 타인.. Locked!!! 
            cv2.putText(image, "U", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)
    except:
        #얼굴 검출 안됨 
        cv2.putText(image, "Face Not Found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(5)==13:
        break
vidcap.release()
cv2.destroyAllWindows()