import dlib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects


detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('./project/models/shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('./project/models/dlib_face_recognition_resnet_model_v1.dat')

def find_faces(img):
    dets = detector(img, 1)

    if len(dets) == 0:
        return np.empty(0), np.empty(0), np.empty(0)


    rects, shapes = [], []
    shapes_np = np.zeros((len(dets), 68, 2), dtype=np.int)
    for k, d in enumerate(dets):
        rect = ((d.left(), d.top()), (d.right(), d.bottom()))
        rects.append(rect)

        shape = sp(img, d)

         
        # 넘파이 배열로 쉐이프 리턴
        for i in range(0, 68):
            shapes_np[k][i] = (shape.part(i).x, shape.part(i).y)

        shapes.append(shape)    
    return rects, shapes, shapes_np

def encode_faces(img, shapes):
    face_descriptors = []
    for shape in shapes:
        face_descriptor = facerec.compute_face_descriptor(img, shape)
        face_descriptors.append(np.array(face_descriptor))

    return np.array(face_descriptors)

 
# 이미지 등록

# find_namelist = np.load('./project/imgtest/find_namelist.npy', allow_pickle=True)

# img_paths = {}
# # descs = {}
# for i in range(len(find_namelist)): 
#     img_paths[i] = './project/imgtest/'+str(i)+ '/' + str(i)+ '.jpg'
#     # descs[i] = None
# print(img_paths)


img_paths = {
    'suzy': './img/suzy.jpg',
    'juhyuk': './img/juhyuk.jpg',
    'hanna': './img/hanna.png',
    'sunho': './img/sunho.png'
}
# print(img_paths.items())
    
# descs = {
#     'neo': None,
#     'trinity': None,
#     'morpheus': None,
#     'smith': None
# }

descs = {
    'suzy': None,
    'juhyuk': None,
    'hanna': None,
    'sunho': None
    # '4': None,
    # '5': None,
    # '6': None,
    # '7': None,
    # '8': None,
    # '9': None,
    # '10': None,
    # '11': None,
    # '12': None,
    # '13': None,
    # '14': None,
    # '15': None,
    # '16': None,
    # '17': None
}

for name, img_path in img_paths.items():
    img_bgr = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    _, img_shapes, _ = find_faces(img_rgb)
    descs[name] = encode_faces(img_rgb, img_shapes)[0]
    print(descs[name])


np.save('./img/startup2.npy', descs)
print(descs)  


# 검증 사진 인풋 부분    
img_bgr = cv2.imread('./img/123.jpg')
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

rects, shapes, _ = find_faces(img_rgb)
descriptors = encode_faces(img_rgb, shapes)

# descs = np.load('./img/descs.npy', allow_pickle=True)
# descs = np.load('./img/descs.npy')

# 결과 출력
fig, ax = plt.subplots(1, figsize=(20, 20))
ax.imshow(img_rgb)

for i, desc in enumerate(descriptors):
    
    found = False
    for name, saved_desc in descs.items():
        dist = np.linalg.norm([desc] - saved_desc, axis=1)

        if dist < 0.6:
            found = True

            text = ax.text(rects[i][0][0], rects[i][0][1], name,
                    color='b', fontsize=40, fontweight='bold')
            text.set_path_effects([path_effects.Stroke(linewidth=10, foreground='white'), path_effects.Normal()])
            rect = patches.Rectangle(rects[i][0],
                                 rects[i][1][1] - rects[i][0][1],
                                 rects[i][1][0] - rects[i][0][0],
                                 linewidth=2, edgecolor='w', facecolor='none')
            ax.add_patch(rect)

            break
    
    if not found:
        ax.text(rects[i][0][0], rects[i][0][1], 'unknown',
                color='r', fontsize=20, fontweight='bold')
        rect = patches.Rectangle(rects[i][0],
                             rects[i][1][1] - rects[i][0][1],
                             rects[i][1][0] - rects[i][0][0],
                             linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

plt.axis('off')
# plt.savefig('./project/result_Img/startup_Output3.png')
plt.show()
