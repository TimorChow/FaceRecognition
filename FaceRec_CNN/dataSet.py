# coding:utf8
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
import random

import os
import cv2
import numpy as np


def read_file(path):
    """
    输入一个文件路径，对其下的每个文件夹下的图片读取，并对每个文件夹给一个不同的Label
    :param path: img路径
    :return: 返回一个img的list,返回一个对应label的list,返回一下有几个文件夹（有几种label)
    """
    img_list = []
    label_list = []
    dir_counter = 0
    IMG_SIZE = 128

    # 对路径下的所有子文件夹中的所有jpg文件进行读取并存入到一个list中
    for child_dir in os.listdir(path):
         child_path = os.path.join(path, child_dir)
         for dir_image in os.listdir(child_path):
             if dir_image.endswith('jpg') or dir_image.endswith('jpeg'):
                img = cv2.imread(os.path.join(child_path, dir_image))
                # 调整大小, 并转成gray
                resized_img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                recolored_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                img_list.append(recolored_img)
                label_list.append(dir_counter)
         dir_counter += 1

    # 返回的img_list转成了 np.array的格式
    img_list = np.array(img_list)

    return img_list, label_list, dir_counter


# 读取训练数据集的文件夹，把他们的名字返回给一个list
def read_name_list(path):
    name_list = []
    for child_dir in os.listdir(path):
        name_list.append(child_dir)
    return name_list


class DataSet(object):
    """
    建立一个用于存储和格式化读取训练数据的类
    """
    def __init__(self,path):
        self.num_classes = None
        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.img_size = 128
        self.extract_data(path)  # 在这个类初始化的过程中读取path下的训练数据

    def extract_data(self, path):
        """
        根据指定路径读取出图片、标签和类别数
        :param path: 数据路径
        :return:
        """
        imgs, labels, counter = read_file(path)

        # 将数据集打乱随机分组
        X_train, X_test, y_train,y_test = train_test_split(imgs, labels, test_size=0.01, random_state=random.randint(0, 100))

        # 数据格式转换, 压缩到0-1之间
        X_train = X_train.reshape(X_train.shape[0], self.img_size, self.img_size, 1)/255.0
        X_test = X_test.reshape(X_test.shape[0],  self.img_size, self.img_size, 1) / 255.0

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        # 将labels转成独热码
        Y_train = np_utils.to_categorical(y_train, num_classes=counter)
        Y_test = np_utils.to_categorical(y_test, num_classes=counter)

        # 将格式化后的数据赋值到属性
        self.X_train = X_train
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_test = Y_test
        self.num_classes = counter

    def check(self):
        print('num of dim:', self.X_test.ndim)
        print('shape:', self.X_test.shape)
        print('size:', self.X_test.size)

        print('num of dim:', self.X_train.ndim)
        print('shape:', self.X_train.shape)
        print('size:', self.X_train.size)


if __name__ == '__main__':
    img_list, label_list, counter = read_file('face')
    print(counter)

