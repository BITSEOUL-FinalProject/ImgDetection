from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization
import PIL.Image as pilimg
from keras import regularizers
import cv2

np.random.seed(33)

# 이미지 생성 옵션 정하기
# train_datagen = ImageDataGenerator(rescale=1./255, #정규화
#                                    horizontal_flip=True, #수평
#                                    vertical_flip=True, #수직
#                                    width_shift_range=0.1, #수평이동
#                                 #    height_shift_range=0.1, #수직이동
#                                    rotation_range=5,
#                                    zoom_range=1.2,
#                                    shear_range=0.7, #좌표보정, 좌표이동
#                                    fill_mode='nearest' #옮겼을때 빈자리를 그전과 비슷하게 채워준다
# )
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True, 
                                #    width_shift_range=0.01,
                                #    height_shift_range=0.01,
                                   rotation_range=5, # 이미지 회전범위 degrees
                                    zoom_range=[0.5,1.0], # 임의 확대/축소 범위
                                #    shear_range=0.7, # 
                                   fill_mode="nearest",
                                   brightness_range=[0.2,1.0])

test_datagen = ImageDataGenerator(rescale=1./255)
pred_datagen = ImageDataGenerator(rescale=1./255.)
train_path   = './teamProject/images_0123/train'
train_g_path = './teamProject/images_0123_g/train'
test_path    = './teamProject/images_0123/test'
test_g_path  = './teamProject/images_0123_g/test'

# 1. 데이터
xy_train = train_datagen.flow_from_directory(
    train_path, #폴더 위치 
    target_size=(200,200), #이미지 크기 설정 - 기존 이미지보다 크게 설정하면 늘려준다 
    batch_size=10, 
    class_mode='categorical', #클래스모드는 찾아보기!
    save_to_dir=train_g_path
) 


# xy_test = test_datagen.flow_from_directory(
#     test_path,
#     target_size=(200,200),
#     batch_size=4,  
#     class_mode='categorical',
#     save_to_dir=test_g_path
# )

# predict = pred_datagen.flow_from_directory(
#     './MJK/data/pred', 
#     target_size=(200,200),
#     batch_size=80,  
#     class_mode=None,
#     shuffle=True
# )
'''
# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization

model = Sequential()
# model.add(Conv2D(64, (2,2), padding='same', activation='linear', input_shape=(200, 200, 3))) 
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Conv2D(128, (2,2), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(Conv2D(64, (2,2), padding='same', activation='linear'))
# model.add(BatchNormalization())
# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=2))
# model.add(Dropout(0.2))
# model.add(Flatten()) 
# model.add(Dropout(0.2))

# model.add(Dense(32, activation='relu'))
# model.add(Dense(4, activation='softmax')) 
model.add(Conv2D(64,(3,3),input_shape=(200,200,3),activation='relu')) 
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
model.add(Dense(1,activation="softmax"))   
model.summary()

#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics =['acc'])
from tensorflow.keras.callbacks import EarlyStopping 
es = EarlyStopping(monitor='val_loss',patience=20, mode='auto')
hist = model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=40, #
    epochs=100,
    validation_data=xy_test, #test도 x, y의 데이터를 모두 가지고 있다
    validation_steps=10, #어떤건지 찾아보기
    callbacks=es
)

#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

# y_pred = model.predict(predict)
# print("y_pred : ", y_pred)
# print(y_pred.shape)
'''