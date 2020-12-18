import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization, Activation
from tensorflow.keras.applications import InceptionResNetV2


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
def hamsu(cnt):
    data_path = './teamProject/images_0123_g3/'+str(cnt)+"/"

    #faces폴더에 있는 파일 리스트 얻기 
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

    #파일 개수 만큼 루프 
    for i, files in enumerate(onlyfiles):    
    
        image_path = data_path + onlyfiles[i]
        #이미지 불러오기 
        # images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        images = cv2.imread(image_path)
        
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
hamsu(cnt+2)
hamsu(cnt+3)

x = np.array(Training_Data)
y = np.array(Labels)
np.save("./data/CV05_2_4_ch3_x_2",arr=x)
np.save("./data/CV05_2_4_ch3_y_2",arr=y)

# x = np.load("./data/CV05_2_4_ch3_x.npy")
# y = np.load("./data/CV05_2_4_ch3_y.npy")

x = x.reshape(x.shape[0],200*200*3)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.8,shuffle=True,random_state=1214)
x_train = x_train.reshape(x_train.shape[0],200,200,3)
x_test = x_test.reshape(x_test.shape[0],200,200,3)

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

# 2. 모델
inceptionResNetV2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(200,200,3))
inceptionResNetV2.trainable = False

model = Sequential()
model.add(inceptionResNetV2)
model.add(Flatten())   
model.add(Dense(100, kernel_initializer='he_normal'))     
model.add(BatchNormalization())
model.add(Activation('relu')) 
model.add(Dense(40, kernel_initializer='he_normal'))                                    
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(4,activation="softmax"))   
model.summary()

# ES
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto')

# LR
lr = ReduceLROnPlateau(monitor='val_loss',
                                patience=3,
                                factor=0.3,
                                verbose=1)
# ModelCheckPoint
modelpath = "./model/CV05_2_4_ch3_MCP_1214-{epoch:02d}-{val_loss:.4f}.hdf5" 
mcp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                      save_best_only=True, mode="auto")

model.compile(loss="categorical_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_train,y_train,epochs=100000,batch_size=64,validation_split=0.5,callbacks=[es,mcp,lr],verbose=1)

# 4. 평가 예측
loss , acc= model.evaluate(x_test,y_test,batch_size=64)

y_pred = model.predict([x_test[0:10]])
aa = np.argmax(y_test[0:10],axis=1).reshape(10)
bb = np.argmax(y_pred,axis=1).reshape(10)
print("aa : ",aa)
print("bb : ",bb)
print("pred : ",y_pred)
print("real : ",y_test[0:10])
