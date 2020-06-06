import numpy as np   # 数据处理的库 numpy
import cv2           # 图像处理的库 OpenCv
import json          # 处理json文件
import time          # 用于测试耗时

from utils.face_utils import face_distance, get_face_encodings, get_face_locations
from utils.db_utils import postgres_db, get_known_id_name_encoding, get_known_id_name_encoding_from_txt

# 全局变量, 待会要写的字体
FONT = cv2.FONT_HERSHEY_DUPLEX


def get_name_id(known_face_encodings, face_encodings):
    """
    线性查找办法, 在给定的known_face_encodings中查找, 与face_encodings欧式距离最近的人脸数据
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
        if min_distance < 0.3:
            name = known_face_names[best_match_index]
            _id = known_face_id[best_match_index]
        face_names.append(name)
        face_id.append(_id)
    return face_names, face_id

"""
LSH查找算法, 精度欠缺
"""
global lsh
lsh = None
def make_lsh():
    from lshash import LSHash
    # 将128D hash映射到k个Bin, 然后在同一Bin中, 寻找最距离最小的
    lsh = LSHash(15, 128)
    results = postgres_db.get_record()
    for i in range(10):
        for re in results:
            encoding = json.loads([re[2]][0])
            lsh.index(encoding)

def lsh_get_name_id(face_encodings):
    """
    根据Lsh算法在库中查找最近邻, 损失精度
    :param face_encodings:
    :return:
    """
    if lsh is None:
        make_lsh()
    face_names = []
    face_id = []
    for encoding in face_encodings:
        lsh_result = lsh.query(encoding, distance_func='true_euclidean')
        name = "Unknown"
        _id = None
        if len(lsh_result) != 0:

            found_encoding = list(lsh_result[0][0])

            info_result = postgres_db.get_info_by_encoding(json.dumps(found_encoding))[0]

            name = info_result[1]
            _id = info_result[0]
        else:
            print('没检测到')
            print(len(lsh_result))
        face_names.append(name)
        face_id.append(_id)

    return face_names, face_id


if __name__ == "__main__":
    # 创建 cv2 摄像头对象
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture("http://admin:admin@192.168.2.126:8081")

    # 将图像缩小, 加快处理, resize为缩小的倍数
    # known_face_id, known_face_names, known_face_encodings, known_face_tmp_id = get_known_id_name_encoding()
    known_face_id, known_face_names, known_face_encodings = get_known_id_name_encoding_from_txt()

    while cap.isOpened():
        flag, img = cap.read()
        kk = cv2.waitKey(1)
        """
        图像预处理
        """
        # 将图像缩小, 加快识别速度
        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)


        """
        获取人脸特征
        """
        # 得到当前帧里的所有人脸
        face_locations = get_face_locations(small_frame)
        # 给定人脸坐标和是原图像, 获取face_encoding
        face_encodings = get_face_encodings(small_frame, face_locations)

        """
        对比识别
        """
        face_names, face_ids = get_name_id(known_face_encodings, face_encodings)
        # face_names, face_ids = lsh_get_name_id(face_encodings)    # 用到了LSH算法


        """
        识别结果处理, 标记到屏幕上
        """
        # 在屏幕上展示识别结果
        # for (top, right, bottom, left), name, _id in zip(face_locations, face_names, face_ids):
        for face, name, _id in zip(face_locations, face_names, face_ids):
            # 之前压缩到0.25, 这里要放大
            top, right, bottom, left = face[0]*4, face[1]*4, face[2]*4, face[3]*4

            # 标记人脸位置
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # 标记人名和ID
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, str(_id) + ':' + name, (left + 6, bottom - 6), FONT, 1.0, (255, 255, 255), 1)


        """
        根据找到的id做事件触发
        """
        # 此处可添加函数做事件触发

        # 窗口显示 show with opencv
        cv2.imshow("camera", img)
        # 按下 q 键退出
        if kk == ord('q'):
            break

    # 释放摄像头 release camera
    cap.release()
    # 删除建立的窗口 delete all the windows
    cv2.destroyAllWindows()
