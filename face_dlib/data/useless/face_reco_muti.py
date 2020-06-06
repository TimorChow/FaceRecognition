import numpy as np  # 数据处理的库 numpy
import cv2  # 图像处理的库 OpenCv
import time
from multiprocessing import Pool, Queue
import queue

from utils.face_utils import face_distance, get_face_encodings, get_face_locations
from utils.db_utils import postgres_db, get_known_id_name_encoding

# 全局变量, 待会要写的字体
FONT = cv2.FONT_HERSHEY_DUPLEX


def get_name_id(face_encodings):
    """
    服务器端工作: 接受摄像头内所有的人脸, 与数据库中的人脸进行对比
    :param face_encodings: 一系列的位置人脸encoding的列表
    :return: 返回所有人脸的id和名字
    """
    # 定义一个新的列表, 储存识别出来的脸对应人名
    face_names = []
    face_id = []

    for encoding in face_encodings:
        # 默认当前人脸为unknow, _id 为 none
        name, _id = 'Unknown', None

        # 遍历人脸, 获取与已知的encoding的欧氏距离, 获取最佳match
        distances = face_distance(known_face_encodings, encoding)
        best_match_index = np.argmin(distances)
        min_distance = distances[best_match_index]
        # 判断是否同一人的欧氏距离阈值, 如果是则更改name 和 _id
        if min_distance < 0.4:
            name = known_face_names[best_match_index]
            _id = known_face_id[best_match_index]
        face_names.append(name)
        face_id.append(_id)
    return face_names, face_id


def process(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    small_frame = img
    small_frame = cv2.resize(small_frame, (0, 0), fx=1 / resize, fy=1 / resize)

    delay_start = time.time()

    """
    获取人脸特征
    """
    # 得到当前帧里的所有人脸
    face_locations = get_face_locations(small_frame)
    face_encodings = get_face_encodings(small_frame, face_locations)

    time_enc = time.time() - delay_start

    """
    对比识别
    """
    face_names, face_ids = get_name_id(face_encodings)

    """
    耗时测试
    """
    time_use = time.time() - delay_start
    count = len(face_locations)
    # 输出encoding消耗时间
    if count != 0:
        count = str(count)
        encoding_time_list[count] = np.append(encoding_time_list[count], time_enc)
        print("Encoding: ", [encoding_time_list[count].mean() for count in ['1', '2', '3']])

    # 输出total消耗时间
    if count != 0:
        count = str(count)
        total_time_list[count] = np.append(total_time_list[count], time_use)
        print("Total Delay: ", [total_time_list[count].mean() for count in ['1', '2', '3']])

    """
    识别结果处理
    """
    # 在屏幕上展示识别结果
    for face, name, _id in zip(face_locations, face_names, face_ids):
        top, right, bottom, left = face[0]*resize, face[1]*resize, face[2]*resize, face[3]*resize

        # 标记人脸位置
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

        # 标记人名和ID
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        cv2.putText(img, str(_id) + ':' + name, (left + 6, bottom - 6), FONT, 1.0, (255, 255, 255), 1)

    IMG = img

    return img


total_time_list = {'1': np.array([0]),
                   '2': np.array([0]),
                   '3': np.array([0])}
encoding_time_list = {'1': np.array([0]),
                      '2': np.array([0]),
                      '3': np.array([0])}

IMG = None

today_known_guest = {}
today_unknown_guest = 0
if __name__ == "__main__":
    # 创建 cv2 摄像头对象
    cap = cv2.VideoCapture(0)

    # 将图像缩小, 加快处理, resize为缩小的倍数
    resize = 4
    known_face_id, known_face_names, known_face_encodings, known_face_tmp_id = get_known_id_name_encoding()

    count = 0
    p = Pool(4)
    q = Queue()
    while cap.isOpened():
        flag, img = cap.read()
        kk = cv2.waitKey(1)

        q.put(img)


        img_labeled = process(img)

        # cv2.waitKey()
        cv2.imshow("camera", IMG)

        cv2.waitKey(1)
        # 按下 q 键退出

        count+=1
        if kk == ord('q'):
            break


    # 释放摄像头 release camera
    cap.release()
    # 删除建立的窗口 delete all the windows
    cv2.destroyAllWindows()
