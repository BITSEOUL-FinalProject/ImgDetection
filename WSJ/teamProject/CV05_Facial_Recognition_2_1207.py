# 리사이즈된 이미지 가져와서 keras 모델 돌려보기


import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

#데이터와 매칭될 라벨 변수 
Training_Data, Labels = [], []

def hamsu(cnt):
    data_path = './teamProject/images3/'+str(cnt)+"/"

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
        Training_Data.append(np.asarray(images, dtype=np.uint8))

        #Labels 리스트엔 카운트 번호 추가 
        Labels.append(cnt)

cnt = 0    
hamsu(cnt)
hamsu(cnt+1)

Training_Data = np.array(Training_Data)
Labels        = np.array(Labels)
print(Training_Data.shape)  
print(Labels.shape)  

Training_Data = Training_Data.reshape(Training_Data.shape[0],200*200)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(Training_Data , Labels , train_size=0.6,shuffle=True,random_state=1207)
x_val, x_test , y_val  , y_test = train_test_split(x_test , y_test , test_size=0.5, shuffle=True,random_state=1207)

x_train = x_train.reshape(x_train.shape[0],50,50,16)
x_test = x_test.reshape(x_test.shape[0],50,50,16)
x_val = x_val.reshape(x_test.shape[0],50,50,16)

print(x_train.shape)
print(x_test.shape)
print(x_val.shape)


# 2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout

model = Sequential()
model.add(Conv2D(10,(3,3),input_shape=(50,50,16),activation='relu')) 
model.add(Flatten())  
model.add(Dense(20,activation='relu'))  
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))                       
model.summary()

# ES
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss',patience=100, mode='auto')

model.compile(loss="binary_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_train,y_train,epochs=10000,batch_size=10,validation_data=(x_val,y_val),callbacks=[es],verbose=1)

# 4. 평가 예측
loss , acc= model.evaluate(x_test,y_test,batch_size=10)

print("loss : ",loss)
print("acc : ",acc)

y_pred = model.predict(x_test)

print("real : ",y_test)
print("pred : ",np.round(y_pred))







'''
#훈련할 데이터가 없다면 종료.
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

#Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)

#모델 생성 
model = cv2.face.LBPHFaceRecognizer_create()

#학습 시작 
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")

'''


















'''
for i, files in enumerate(onlyfiles):    
    image_path = data_path + onlyfiles[i]
    #이미지 불러오기 
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    #이미지 파일이 아니거나 못 읽어 왔다면 무시
    if images is None:
        continue    
    #Training_Data 리스트에 이미지를 바이트 배열로 추가 
    Training_Data.append(np.asarray(images, dtype=np.uint8))

    #Labels 리스트엔 카운트 번호 추가 
    Labels.append(i)

#훈련할 데이터가 없다면 종료.
if len(Labels) == 0:
    print("There is no data to train.")
    exit()

#Labels를 32비트 정수로 변환
Labels = np.asarray(Labels, dtype=np.int32)

#모델 생성 
model = cv2.face.LBPHFaceRecognizer_create()

#학습 시작 
model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Model Training Complete!!!!!")
'''