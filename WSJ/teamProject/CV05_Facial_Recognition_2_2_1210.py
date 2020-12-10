import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout, BatchNormalization

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

x = np.append(Training_Data, Training_Data2, axis=0)
y = np.append(Labels, Labels2, axis=0)

x = x.reshape(x.shape[0],200*200)

from sklearn.model_selection import train_test_split
x_train, x_test , y_train  , y_test = train_test_split(x , y , train_size=0.6,shuffle=True,random_state=14)
x_train = x_train.reshape(x_train.shape[0],100,100,4)
x_test = x_test.reshape(x_test.shape[0],100,100,4)

# from tensorflow.keras.utils import to_categorical

# y_train = to_categorical(y_train)
# y_test  = to_categorical(y_test)

# 2. 모델


model = Sequential()
model.add(Conv2D(64,(3,3),input_shape=(100,100,4),activation='relu')) 
model.add(BatchNormalization())

model.add(Conv2D(64,(3,3),activation='relu'))     
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2     

model.add(Conv2D(128,(3,3),activation='relu'))                      
model.add(BatchNormalization())

model.add(Conv2D(128,(6,6),activation='relu'))                      
model.add(BatchNormalization())
model.add(Dropout(0.2))                
model.add(MaxPooling2D(pool_size=2))   # pool_size default : 2  

model.add(Conv2D(256,(3,3),activation='relu')) 
model.add(BatchNormalization())


model.add(Conv2D(256,(3,3),activation='relu')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(512,(3,3),activation='relu')) 
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Flatten())    
model.add(Dense(1,activation="sigmoid"))                       
model.summary()

# import tensorflow as tf
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(100, 100, 4)),
#     tf.keras.layers.MaxPooling2D(2, 2),  
#     tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(256, (6, 6), padding="same", activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(512, activation="relu"),
#     tf.keras.layers.Dense(2, activation='softmax')
# ])

# ES
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto')

# LR
lr = ReduceLROnPlateau(monitor='val_loss',
                                patience=3,
                                factor=0.3,
                                verbose=1)
# ModelCheckPoint
modelpath = "./model/CV05_2_2_MCP_1212-{epoch:02d}-{val_loss:.4f}.hdf5" 
mcp = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                      save_best_only=True, mode="auto")

model.compile(loss="binary_crossentropy",optimizer="adam",metrics="acc")
model.fit(x_train,y_train,epochs=10000,batch_size=32,validation_split=0.5,callbacks=[es,mcp,lr],verbose=1)

# 4. 평가 예측
loss , acc= model.evaluate(x_test,y_test,batch_size=32)

# y_pred = model.predict(x_test[:10])
# print(np.argmax(y_pred))