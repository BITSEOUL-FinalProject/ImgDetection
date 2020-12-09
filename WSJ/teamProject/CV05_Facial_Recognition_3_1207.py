# 종합
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

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




#데이터와 매칭될 라벨 변수 

Training_Data, Labels = [], []
Training_Data2, Labels2 = [], []
def hamsu(cnt,t):
    print(cnt)
    data_path = './teamProject/images5/'+str(cnt)+"_"+str(t)+"/"

    #faces폴더에 있는 파일 리스트 얻기 
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    #파일 개수 만큼 루프 
    for i, files in enumerate(onlyfiles):    
    
        image_path = data_path + onlyfiles[i]
        #이미지 불러오기 
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #이미지 파일이 아니거나 못 읽어 왔다면 무시
        if images is None:
            continue    
        #Training_Data 리스트에 이미지를 바이트 배열로 추가 
        
        if t == "test":
            Training_Data2.append(np.asarray(images, dtype=np.uint8))
            Labels2.append(cnt)

        elif t == "train":
            Training_Data.append(np.asarray(images, dtype=np.uint8))
            Labels.append(cnt)
          

cnt = 0    
hamsu(cnt,"train")
hamsu(cnt,"test")
hamsu(cnt+1,"train")
hamsu(cnt+1,"test")

Training_Data2 = np.array(Training_Data2)
Labels2        = np.array(Labels2)
Training_Data = np.array(Training_Data)
Labels        = np.array(Labels)

# print(Training_Data.shape)  
# print(Labels.shape)  
# print(Training_Data2.shape)  
# print(Labels2.shape)  

# x_train = Training_Data
# x_test  = Training_Data2
# y_train = Labels
# y_test  = Labels2

# x_train = x_train.reshape(x_train.shape[0],100,100,4)
# x_test = x_test.reshape(x_test.shape[0],100,100,4)
x = np.append(Training_Data, Training_Data2, axis=0)
y = np.append(Labels, Labels2, axis=0)

x = x.reshape(x.shape[0],200*200)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8,shuffle=True,random_state=13)
x_train = x_train.reshape(x_train.shape[0],100,100,4)
x_test = x_test.reshape(x_test.shape[0],100,100,4)

# x_val, x_test , y_val  , y_test = train_test_split(x_test , y_test , test_size=0.5, shuffle=True,random_state=12)

# x_train = x_train.reshape(x_train.shape[0],50,50,16)
# x_test = x_test.reshape(x_test.shape[0],50,50,16)
# x_val = x_val.reshape(x_val.shape[0],50,50,16)

# print(x_train.shape)
# print(x_test.shape)
# print(x_val.shape)



# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization

model = Sequential()
# model.add(Conv2D(10,(3,3),input_shape=(50,50,16),activation='relu')) 
# model.add(Flatten())  
# model.add(Dense(20,activation='relu'))  
# model.add(Dense(10,activation='relu'))
model.add(Conv2D(64,(3,3),input_shape=(100,100,4),activation='relu')) 
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu'))     
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2     

model.add(Conv2D(128,(3,3),activation='relu'))                      
model.add(BatchNormalization())

model.add(Conv2D(128,(3,3),activation='relu'))                      
model.add(BatchNormalization())
model.add(Dropout(0.2))                
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2  

model.add(Conv2D(256,(3,3),activation='relu')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Flatten())     
# model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation="sigmoid"))                       
model.summary()

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=10, mode='auto')

model.compile(loss="binary_crossentropy",optimizer="adam",metrics="acc")



# from xgboost import XGBClassifier, XGBRegressor, plot_importance
# from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
# from sklearn.model_selection import GridSearchCV , RandomizedSearchCV
# def create_hyperparameter():
#     # batches = [10, 20, 30, 40, 50]
#     # optimizers = ['rmsprop', 'adam', 'adadelta']
#     # dropout = np.linspace(0.1, 0.5, 5)
#     batches = [50]
#     optimizers = ['adam']
#     # dropout = np.linspace(0.1,0.5, 5)
#     return{}

# hyperparameters = create_hyperparameter()

# model = KerasClassifier(build_fn=build_model, verbose=1)


# model = GridSearchCV(model,hyperparameters,cv=3,verbose=1)
# model = XGBClassifier(model,cv=5)
# model.fit(x_train,y_train,verbose=True, eval_metric="error", 
#           eval_set=[(x_train,y_train),(x_test,y_test)],early_stopping_rounds=30)

model.fit(x_train,y_train,epochs=1,batch_size=32,validation_split=0.5,callbacks=[es],verbose=1)

# # 4. 평가 예측
loss , acc= model.evaluate(x_test,y_test,batch_size=32)
# 329/329 [==============================] - 7s 20ms/step - loss: 0.0853 - acc: 0.9692 - val_loss: 0.1776 - val_acc: 0.9389
# 165/165 [==============================] - 1s 5ms/step - loss: 0.1856 - acc: 0.9330
# result = model.predict(x_test)
# r = np.argmax(result[1])
# print(r)
# ==================================================================================

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

#파일 경로
FilePath = './teamProject/video/startup.mp4'

#Open the File
movie = cv2.VideoCapture(FilePath) #동영상 핸들 얻기

#Check that the file is opened
if movie.isOpened() == False: #동영상 핸들 확인
    print('Can\'t open the File' + (FilePath))
    exit()

#create the window & change the window size
#윈도우 생성 및 사이즈 변경
# cv2.namedWindow('Face')


while True:
    #카메라로 부터 사진 한장 읽기 
    ret, frame = movie.read()

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
movie.release()
cv2.destroyAllWindows()

