import CV07_CVlib_test
import CV07_Dlib_test
import CV07_OpenCV_test
from tensorflow.keras.models import load_model
from cv2 import cv2
import math
import time as t

def dd(name,time):

    # cvlib
    cvlib_model_path = './model/rmsprop_best.hdf5'
    cvlib_model = load_model(cvlib_model_path)
    cvlib_character = CV07_CVlib_test.character()

    # openCV
    OpenCV_model_path = "./model/CV05_2_3_MCP_1214-23-3.8027.hdf5"
    OpenCV_model = load_model(OpenCV_model_path)
    face_classifier = CV07_OpenCV_test.face_classifier()
    
    # dlib
    dlib_data = CV07_Dlib_test.load_ImageData()
    dlib_emptyList = CV07_Dlib_test.make_Emptylist()

    # video
    videoPath = "./teamProject/video/"+name
    video     = cv2.VideoCapture(videoPath)

    width  = video.get(cv2.CAP_PROP_FRAME_WIDTH) # 또는 movie.get(3)
    height = video.get(cv2.CAP_PROP_FRAME_HEIGHT) # 또는 movie.get(4)
    fps    = video.get(cv2.CAP_PROP_FPS)

    time     = int(time)*int(math.ceil(24))
    count    = 0


    op_save_path = './teamProject/static/op3_'+name
    cv_save_path = './teamProject/static/cv3_'+name
    dl_save_path = './teamProject/static/dl3_'+name

    fourcc = cv2.VideoWriter_fourcc('H','2','6','4') 
    # write1 = cv2.VideoWriter(op_save_path, fourcc, 24, (int(width), int(height)))
    write2 = cv2.VideoWriter(cv_save_path, fourcc, 24, (int(width), int(height)))
    # write3 = cv2.VideoWriter(dl_save_path, fourcc, 24, (int(width), int(height)))

    start = t.time()
    print("start : ",start)

    while video.isOpened():
        # 인식 및 검출
        cvlib_frame  = CV07_CVlib_test.CVlib(video, cvlib_model, cvlib_character)
        # dlib_frame   = CV07_Dlib_test.Dlib(video, dlib_emptyList, dlib_data)
        # opencv_frame = CV07_OpenCV_test.OpenCV(video, OpenCV_model)

        # 리사이즈
        # cvlib_frame  = cv2.resize(cvlib_frame, (720, 480))
        # dlib_frame   = cv2.resize(dlib_frame, (720, 480))
        # opencv_frame = cv2.resize(opencv_frame, (720, 480))

        # write
        cv_count=CV07_CVlib_test.videoWrite(write2,cvlib_frame)
        # dl_count=CV07_Dlib_test.videoWrite(write3,dlib_frame)
        # op_count=CV07_OpenCV_test.videoWrite(write1,opencv_frame)

        # show
        CV07_CVlib_test.cvlib_show(cvlib_frame)
        # CV07_Dlib_test.Dlib_show(dlib_frame)
        # CV07_OpenCV_test.OpenCV_show(opencv_frame)

        count = count+1
        if count == time:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    end = t.time()
    print("end : ",end)

    # write1.release()
    write2.release()
    # write3.release()
    video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    dd("video2.mp4",30)
