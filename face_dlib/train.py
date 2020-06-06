import os
import time
import cv2

from utils.db_utils import postgres_db, save_to_txt
from utils.face_utils import get_face_encodings


def train():
    known_face_encodings = []
    known_face_names = []

    IMAGE_PATH = "src/"
    count = 0
    # 遍历图片文件提取
    for name in os.listdir(IMAGE_PATH):
        # 过滤无关文件
        if ('features' in name) or ('name' in name) or ('Store' in name):
            continue
        user_image = cv2.imread(IMAGE_PATH + name)

        """
        获取人脸特征
        """
        encoding = get_face_encodings(user_image)
        # 如果没有识别出人脸, 跳过
        if len(encoding) == 0: continue

        """
        Name和Encoding赋值
        """
        encoding = encoding[0]
        user_name = name.split('.')[0]

        known_face_names.append(user_name)
        known_face_encodings.append(encoding)

        """
        进度输出
        """
        count += 1
        if count % 100 == 0:
            print(count)

    """
    数据存储, 可以选择数据库或文件
    """
    _id = 0
    for name, encoding in zip(known_face_names, known_face_encodings):
        # 将结果存入数据库
        # postgres_db.save_record_to_db(_id, name, encoding)

        # 将结果存入本地文件
        save_to_txt(_id, name, encoding)

        _id += 1

    print('训练完毕')


if __name__ == "__main__":
    start_time = time.time()
    train()
    end_time = time.time()
    print(end_time - start_time)

    postgres_db.close()
