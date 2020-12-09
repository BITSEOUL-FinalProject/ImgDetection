from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionResNetV2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, MaxPooling2D, Flatten, BatchNormalization
import PIL.Image as pilimg
from keras import regularizers
import cv2

# find_namelist = np.load('./Common_data/npy/find_namelist.npy', all)
np.random.seed(33)

# 이미지 생성 옵션 정하기
train_datagen = ImageDataGenerator(rescale=1./255, #정규화
                                   horizontal_flip=True, #수평
                                   vertical_flip=True, #수직
                                   width_shift_range=0.1, #수평이동
                                   height_shift_range=0.1, #수직이동
                                   rotation_range=10,
                                   zoom_range=1.2,
                                   shear_range=0.6, #좌표보정, 좌표이동
                                   fill_mode='nearest' #옮겼을때 빈자리를 그전과 비슷하게 채워준다
)

test_datagen = ImageDataGenerator(rescale=1./255)
pred_datagen = ImageDataGenerator(rescale=1./255.)

# 1. 데이터
xy_train = train_datagen.flow_from_directory(
    './MJK/data/train', #폴더 위치 
    target_size=(200,200), 
    batch_size=10, 
    class_mode='binary' 
    # save_to_dir='./data/img/data1_2/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 

xy_test = test_datagen.flow_from_directory(
   './MJK/data/test',
    target_size=(200,200),
    batch_size=3,  
    class_mode='binary'
    # save_to_dir='./data/img/data1_2/test'
)

predict = pred_datagen.flow_from_directory(
    './MJK/data/pred', 
    target_size=(200,200),
    batch_size=50,  
    class_mode=None,
    shuffle=True
)

# 2. 모델 구성
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout, Activation, BatchNormalization

inceptionResNetV2 = InceptionResNetV2(weights='imagenet', include_top=False, input_shape=(xy_train[0][0].shape[1], xy_train[0][0].shape[2], xy_train[0][0].shape[3]))
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
model.add(Dense(1, activation = 'sigmoid'))

#3. 컴파일, 훈련
modelpath = "D:/ImgDetection/MJK/data/weight/cp-{epoch:002d}-{val_loss: 4f}.hdf5"  

model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['acc'])
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 20)
check_point = ModelCheckpoint(filepath = modelpath, monitor = 'val_loss', save_best_only = True, mode = 'val_loss')
hist = model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=38, #
    epochs=100,
    validation_data=xy_test, #test도 x, y의 데이터를 모두 가지고 있다
    validation_steps=20, #어떤건지 찾아보기
    callbacks=[early_stopping, check_point]
)

#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict_generator(predict, steps=1, verbose=1)
print("y_pred : ", y_pred)
print(y_pred.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
rows = 7
cols = 8

# print(y_pred[1][0])
a = ['B', 'N']

def printIndex(array, i):
    if np.round(array[i][0]) == 0:
        return a[0]
    elif np.round(array[i][0]) == 1:
        return a[1]

for i in range(len(predict[0])):
    ax = fig.add_subplot(rows, cols, i+1)
    ax.imshow(predict[0][i])
    label = printIndex(y_pred, i)
    # print(label)
    ax.set_xlabel(label)
    ax.set_xticks([]), ax.set_yticks([])
plt.show()

# loss :  0.14220841228961945
# acc :  0.949999988079071
# [[0.00511569]
#  [0.7673361 ]
#  [0.97431016]
#  [0.01020213]
#  [0.9927758 ]
#  [0.7497511 ]
#  [0.7221572 ]
#  [0.99856645]
#  [0.05838428]
#  [0.66237336]
#  [0.78285784]
#  [0.9205434 ]
#  [0.03582747]
#  [0.07276288]
#  [0.87418294]
#  [0.99428517]
#  [0.01810936]
#  [0.9824897 ]
#  [0.7433762 ]
#  [0.37383893]
#  [0.00172315]
#  [0.83169407]
#  [0.9973502 ]
#  [0.6480843 ]
#  [0.84695905]
#  [0.00344487]
#  [0.8763713 ]
#  [0.9620815 ]
#  [0.0057925 ]
#  [0.01655022]
#  [0.9447976 ]
#  [0.02900423]
#  [0.9586486 ]
#  [0.01404348]
#  [0.01092414]
#  [0.99086505]
#  [0.05920314]
#  [0.9963148 ]
#  [0.05133378]
#  [0.97355723]
#  [0.4944659 ]
#  [0.74268365]
#  [0.00289558]
#  [0.05846003]
#  [0.99600464]
#  [0.9975501 ]
#  [0.9244332 ]
#  [0.99559987]
#  [0.5892266 ]
#  [0.97763604]]
# (50, 1)