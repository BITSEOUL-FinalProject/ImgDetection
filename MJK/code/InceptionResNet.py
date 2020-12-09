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
    target_size=(200,200), 
    batch_size=19, 
    class_mode='binary' 
    # save_to_dir='./data/img/data1_2/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 

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
model.add(Dense(100, activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Layer (type)                 Output Shape              Param #
# =================================================================
# inception_resnet_v2 (Functio (None, 4, 4, 1536)        54336736
# _________________________________________________________________
# flatten (Flatten)            (None, 24576)             0
# _________________________________________________________________
# dense (Dense)                (None, 100)               2457700
# _________________________________________________________________
# dense_1 (Dense)              (None, 1)                 101
# =================================================================
# Total params: 56,794,537
# Trainable params: 2,457,801
# Non-trainable params: 54,336,736
# _________________________________________________________________


#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics =['acc'])
from tensorflow.keras.callbacks import EarlyStopping 

hist = model.fit_generator(
    xy_train, #train 안에 x, y의 데이터를 모두 가지고 있다
    steps_per_epoch=20, #
    epochs=30,
    validation_data=xy_test, #test도 x, y의 데이터를 모두 가지고 있다
    validation_steps=3 #어떤건지 찾아보기
)

#4. 평가, 예측
loss, acc = model.evaluate(xy_test)
print("loss : ", loss)
print("acc : ", acc)

y_pred = model.predict(predict)
print("y_pred : ", y_pred.reshape(50,))
print(y_pred.shape)

import matplotlib.pyplot as plt

fig = plt.figure()
rows = 5
cols = 11

# print(y_pred[1][0])
a = ['SUJI', 'JOOHYUK']

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

