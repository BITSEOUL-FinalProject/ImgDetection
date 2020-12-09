import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine
from cv2 import cv2
import mtcnn
from keras.models import load_model
sys.path.append('..')
from utils import get_face, l2_normalizer, normalize, save_pickle, plt_show, get_encode

# 경로 설정
encoder_model = 'D:/ImgDetection/KHL/Facenet/data/model/facenet_keras.h5'
people_dir = 'D:/ImgDetection/KHL/Facenet/data/people'
encodings_path = 'D:/ImgDetection/KHL/Facenet/data/encodings/encodings.pkl'
test_img_path = 'D:/ImgDetection/KHL/Facenet/data/test/friends.jpg'
test_res_path = 'D:/ImgDetection/KHL/Facenet/data/results/friends.jpg'

# 변수 설정
recognition_t = 0.3
required_size = (160, 160)
encoding_dict = dict()

# 모델
face_detector = mtcnn.MTCNN()
face_encoder = load_model(encoder_model)

face_encoder.summary()
# for person_name in os.listdir(people_dir):
#     person_dir = os.path.join(people_dir, person_name)
#     encodes = []
#     for img_name in os.listdir(person_dir):
#         img_path = os.path.join(person_dir, img_name)
#         img = cv2.imread(img_path)
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         results = face_detector.detect_faces(img_rgb)       # 얼굴 box 안에있는 양쪽 눈, 입의 왼쪽, 입의 오른쪽 좌표를 받아온다.
#         print(results)
#         if results:
#             res = max(results, key=lambda b: b['box'][2] * b['box'][3])
#             face, _, _ = get_face(img_rgb, res['box'])
            
#             face = normalize(face)
#             face = cv2.resize(face, required_size)
#             encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
#             encodes.append(encode)
#     if encodes:
#         encode = np.sum(encodes, axis=0)
#         encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
#         encoding_dict[person_name] = encode

# for key in encoding_dict.keys():
#     print(key)

# with open(encodings_path, 'bw') as file:
#     pickle.dump(encoding_dict, file)