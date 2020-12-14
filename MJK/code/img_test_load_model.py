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
train_datagen = ImageDataGenerator(     
    rescale=1 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'    
)

test_datagen = ImageDataGenerator(rescale=1./255)
pred_datagen = ImageDataGenerator(rescale=1./255.)

# 1. 데이터
xy_train = train_datagen.flow_from_directory(
    './MJK/data/train', #폴더 위치 
    target_size=(200,200), #이미지 크기 설정 - 기존 이미지보다 크게 설정하면 늘려준다 
    batch_size=64, 
    class_mode='binary' #클래스모드는 찾아보기!  
    # save_to_dir='./data/img/data1_2/train' #전환된 이미지 데이터 파일을 이미지 파일로 저장
) 

# x=(150,150,1), train 폴더안에는 ad/normal이 들어있다. y - ad:0, normal:1

xy_test = test_datagen.flow_from_directory(
   './MJK/data/test',
    target_size=(200,200),
    batch_size=64,  
    class_mode='binary'   
    # save_to_dir='./data/img/data1_2/test'
)

predict = pred_datagen.flow_from_directory(
    './MJK/data/pred', 
    target_size=(200,200),
    batch_size=50,  
    class_mode=None,
    shuffle=False,
)

# 2. 모델 구성
model = load_model('./MJK/data/weight/cp_inc-33-0.095160.hdf5') 

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
    # print(label)
    ax.set_xlabel(label)
    ax.set_xticks([]), ax.set_yticks([])
plt.show()

# loss :  0.16194841265678406
# acc :  0.9437500238418579
#  [0.99698895]
#  [0.02451021]
#  [0.9973544 ]
#  [0.9924245 ]
#  [0.0570064 ]
#  [0.9040135 ]
#  [0.9461852 ]
#  [0.01738477]
#  [0.85327214]
#  [0.99225265]
#  [0.67492515]
#  [0.9836431 ]
#  [0.00730109]
#  [0.07669482]
#  [0.8139179 ]
#  [0.99146324]
#  [0.0135073 ]
#  [0.0036249 ]
#  [0.9564992 ]
#  [0.89642656]
#  [0.6596768 ]
#  [0.5442315 ]
#  [0.4975191 ]
#  [0.05730857]
#  [0.00679069]
#  [0.99110526]
#  [0.04862578]
#  [0.0539133 ]
#  [0.00537304]
#  [0.99581003]
#  [0.9465147 ]
#  [0.01135786]
#  [0.27201033]
#  [0.37967515]
#  [0.75213146]
#  [0.7929427 ]
#  [0.9858181 ]
#  [0.00963665]
#  [0.0036931 ]
#  [0.0090879 ]
#  [0.5691298 ]
#  [0.9956832 ]
#  [0.9988802 ]
#  [0.96950173]
#  [0.98985684]
#  [0.98529667]
#  [0.01022546]
#  [0.9978059 ]
#  [0.03624589]]