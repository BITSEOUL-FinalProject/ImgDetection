from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
np.random.seed(26)
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




data_path_2 = './teamProject/images4/1/test'
test_path  = './teamProject/images5/1_test'
# train_len = len(train_datagen)
# test_len = len(test_datagen)
# for j in range(3): 
#     data_path_1 = './teamProject/images_0123_o/'+str(j)
#     save_path   = './teamProject/images_0123_g3/'+str(j)
#     for i in range(aa):
#         train_generator = train_datagen.flow_from_directory(
#             data_path_1,
#             target_size=(200,200),
#             batch_size=1000, # image를 5장씩
#             class_mode="categorical",
#             shuffle=True,
#             save_to_dir=save_path
#         )
#         next(train_generator)


data_path_1 = './teamProject/images_0123_o/3'
save_path   = './teamProject/images_0123_g3/3'
for i in range(22):
    train_generator = train_datagen.flow_from_directory(
        data_path_1,
        target_size=(200,200),
        batch_size=1000, # image를 5장씩
        class_mode="categorical",
        shuffle=True,
        save_to_dir=save_path
    )
    next(train_generator)   

    
# test_generator = test_datagen.flow_from_directory(
#     data_path_2,
#     target_size=(200,200),
#     batch_size=100, # image를 5장씩
#     class_mode="binary",
#     shuffle=True,
#     save_to_dir=test_path
# )
# next(test_generator)

# model = Sequential()
# model.add(Conv2D(10,(4,4),input_shape=(150,150,3))) 
# model.add(Conv2D(10,(3,3)))                      
# model.add(Conv2D(10,(3,3)))                                      
# model.add(Conv2D(10,(2,2)))                            
# model.add(MaxPooling2D(pool_size=2))   
# model.add(Flatten())                                             
# model.add(Dense(10,activation='relu'))                           
# model.add(Dense(1,activation="sigmoid"))                       
# model.summary()


# model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
# model.fit_generator(
#     train_generator,
#     steps_per_epoch  = 50,
#     epochs           = 200,
#     validation_data  = test_generator,
#     validation_steps = 4
# )


# lentrain = len(train_generator)
# lentest = len(test_generator)
# print(len) # 348

# train_generator = train_datagen.flow_from_directory(
#     "./data/data2",
#     target_size=(200,200),
#     batch_size=5*lentrain, # image를 5장씩
#     class_mode="binary"
#     ,save_to_dir=train_path
# )
# test_generator = test_datagen.flow_from_directory(
#     "./data/data2",
#     target_size=(200,200),
#     batch_size=5*lentest, # image를 5장씩
#     class_mode="binary"
#     ,save_to_dir=test_path
# )