import cv2
import numpy as np
import matplotlib.pyplot as plt

# show했을때 이미지 제외 배경색 바꾸기
plt.style.use("dark_background")


# 인풋이미지 읽기
img_car = cv2.imread("./teamProject/images/test4.jpg")
print(img_car.shape) # (626, 940, 3)

# 이미지의 높이, 너비, 채널을 구한다.
height, width, channel = img_car.shape

# plt.figure(figsize=(12,10))
# plt.imshow(img_car,cmap="gray")
# cmap="gray" 해도 회색안됨
# plt.show()


gray_img = cv2.cvtColor(img_car,cv2.COLOR_BGR2GRAY)
# 이미지를 그레이색으로 바꾸기
# 이미지를 그래이색으로 바꾸더라도 show에서 cmap=gray를 안하면 회색으로 안나옴
# print(gray_img.shape) # (626, 940)
# plt.figure(figsize=(12,10))
# plt.imshow(gray_img,cmap="gray") 
# 여기서 cmap=gray하면 회색으로 변함
# plt.show()

# 쓰레쉬홀딩 할꺼임
# Adaptive Thresholding

img_blurred = cv2.GaussianBlur(gray_img,ksize=(5,5),sigmaX=0)
# 가우시안 필터를 사용하여 이미지를 블러싱한다.
# 쓰레쉬홀딩시 노이즈를 줄여준다.
# 블러처리를 하여 thresholding을 할때 이미지 구분에 유리해짐
# ksize : 를 크게하면할수록 흐려짐
# sigmaX : X 방향의 표준 편차; 0이면 커널 크기에서 계산
# sigmaY : Y 방향의 표준 편차; sigmaY가 None이면 sigmaY는 sigmaX와 같게됩니다.

# plt.figure(figsize=(12,10))
# plt.imshow(img_blurred,cmap="gray") 
# plt.show()

img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=701,
    C=7
    # maxValue : thresholding한 결과를 흰색으로 설정할 때 사용하는 값
    # adaptiveMethod : 주위 픽셀값들의 평균을 구할 때 사용하는 방법
    # thresholdType : thresholding방법을 지정하는 것으로 THRESH_BINARY 혹은
    #                 THRESH_BINARY_INV 둘 중 하나가 되어야 한다.
    #                 THRESH_BINARY_INV는 THRESH_BINARY가 할당한 값을 반대로 하는것이다.
    # blockSize : 주위 블럭들의 범위크기/ 
    #             너무 작은숫자로하면 흰색이 적어지고 반대로하면 너무 많아짐
    # C : 상수로서 평균값과 차이를 구하기 전에 빼는 값으로 보정을 위해 사용된다.
    #     너무 작은숫자로하면 ㅎ반대로하면 너무 많아짐
)
# 검은색은 흰색으로 바꾸고 나머지는 검은색으로 바꿔줌
# plt.figure(figsize=(12,10))
# plt.imshow(img_thresh,cmap="gray") 
# plt.show()

contours,_ = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
    # contours : 윤곽선
    # findContours()는 원본이미지를 직접 수정한다.
    # img는 binary이미지를 사용해야한다.
    # mode : contours를 찾는 방법
    # method : contours찾을때 사용하는 근사치 방법
    # returns : image, contours, hierachy
)

temp_result = np.zeros((height,width,channel),dtype=np.uint8)
# np.zeros : 전체가 0으로 채워저있는 리스트를 반환한다.  
# uint8 = 0~255

cv2.drawContours(temp_result,contours=contours,contourIdx=-1,color=(255,255,255))
# fineContours 로 찾은 윤곽선을 drawContours로 그린다.
# contourIdx = -1 모든 윤곽선을 그리겠다


plt.figure(figsize=(12,10))
plt.imshow(temp_result,cmap="gray") 
plt.show()

'''
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
    
    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
MIN_AREA = 80
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']
    
    if area > MIN_AREA \
    and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)
        
# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
#     cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')


MAX_DIAG_MULTIPLYER = 5 # 5
MAX_ANGLE_DIFF = 12.0 # 12.0
MAX_AREA_DIFF = 0.5 # 0.5
MAX_WIDTH_DIFF = 0.8
MAX_HEIGHT_DIFF = 0.2
MIN_N_MATCHED = 3 # 3

def find_chars(contour_list):
    matched_result_idx = []
    
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)
        
        # recursive
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx
    
result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
#         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)

plt.figure(figsize=(12, 10))
plt.imshow(temp_result, cmap='gray')
plt.show()

PLATE_WIDTH_PADDING = 1.3 # 1.3
PLATE_HEIGHT_PADDING = 1.5 # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
    
    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
    
    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
    
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )
    
    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
    
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
    
    img_rotated = cv2.warpAffine(img_thresh, M=rotation_matrix, dsize=(width, height))
    
    img_cropped = cv2.getRectSubPix(
        img_rotated, 
        patchSize=(int(plate_width), int(plate_height)), 
        center=(int(plate_cx), int(plate_cy))
    )
    
    if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
        continue
    
    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })
    
    plt.subplot(len(matched_result), 1, i+1)
    plt.imshow(img_cropped, cmap='gray')
    plt.show()

'''










