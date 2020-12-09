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
train_datagen = ImageDataGenerator(rescale=1./255, #정규화
                                   horizontal_flip=True, #수평
                                   vertical_flip=True, #수직
                                   width_shift_range=0.1, #수평이동
                                   height_shift_range=0.1, #수직이동
                                   rotation_range=5,
                                   zoom_range=1.2,
                                   shear_range=0.7, #좌표보정, 좌표이동
                                   fill_mode='nearest' #옮겼을때 빈자리를 그전과 비슷하게 채워준다
)

test_datagen = ImageDataGenerator(rescale=1./255)
pred_datagen = ImageDataGenerator(rescale=1./255.)

# 1. 데이터
xy_train = train_datagen.flow_from_directory(
    './MJK/data/train', #폴더 위치 
    target_size=(200,200), #이미지 크기 설정 - 기존 이미지보다 크게 설정하면 늘려준다 
    batch_size=19, 
    class_mode='binary' #클래스모드는 찾아보기!
    # save_to_dir='./data/img/data1_2/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 

# x=(150,150,1), train 폴더안에는 ad/normal이 들어있다. y - ad:0, normal:1

xy_test = test_datagen.flow_from_directory(
   './MJK/data/test',
    target_size=(200,200),
    batch_size=4,  
    class_mode='binary'
    # save_to_dir='./data/img/data1_2/test'
)

predict = pred_datagen.flow_from_directory(
    './MJK/data/pred', 
    target_size=(200,200),
    batch_size=80,  
    class_mode=None,
    shuffle=True
)
print(predict[0][0][1].shape)


# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization

model = Sequential()
model.add(Conv2D(64, (2,2), padding='same', activation='linear', input_shape=(200, 200, 3))) 
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(128, (2,2), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv2D(64, (2,2), padding='same', activation='linear'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten()) 
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid')) 

model.summary()

#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['acc'])
from tensorflow.keras.callbacks import EarlyStopping 

hist = model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=40, #
    epochs=100,
    validation_data=xy_test, #test도 x, y의 데이터를 모두 가지고 있다
    validation_steps=10 #어떤건지 찾아보기
)

#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(predict)
print("y_pred : ", y_pred)
print(y_pred.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
rows = 5
cols = 11

# print(y_pred[1][0])
a = ['BAE', 'NAM']

def printIndex(array, i):
    if np.round(array[i][0]) == 0:
        return a[0]
    elif np.round(array[i][0]) == 1:
        return a[1]

for i in range(len(predict[0])):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(predict[0][i])
    label = printIndex(y_pred, i)
    print(label)
    ax.set_xlabel(label)
    ax.set_xticks([]), ax.set_yticks([])
plt.show()
