# coding:utf8
from dataSet import DataSet
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten, Dropout
import numpy as np


class Model(object):
    FILE_PATH = "model\model.h5"  # 模型存储路径
    IMAGE_SIZE = 128    # 128*128的人脸图

    def __init__(self):
        self.model = None

    def load_trainData(self, dataset):
        """
        读取实例化后的DataSet类作为进行训练的数据源
        :param dataset: object of DataSet Class
        :return:
        """
        self.dataset = dataset

    def build_model(self):
        """
        建立一个CNN模型，一层卷积、一层池化、一层卷积、一层池化、展开之后进行全链接、最后进行分类
        conv + pooling + conv + pooling + dense
        :return:
        """
        self.model = Sequential()

        # 第一层 conv + pooling
        self.model.add(
            Convolution2D(
                filters=32,
                kernel_size=(5, 5),
                padding='same',
                dim_ordering='th',
                input_shape=self.dataset.X_train.shape[1:]
            )
        )
        self.model.add(Activation('relu'))
        self.model.add(
            MaxPooling2D(
                pool_size=(2, 2),
                strides=(2, 2),
                padding='same'
            )
        )

        # 第二层卷积
        self.model.add(Convolution2D(filters=64, kernel_size=(5, 5), padding='same'))
        # 加一层激活函数
        self.model.add(Activation('relu'))  # rectified linear unit
        # pooling 提取特征, 缩小图片, 提高训练速度
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

        # 展开后, 添加全连接层
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(Activation('relu'))

        # 最后分类, 输出层进行分类, 加softmax 压缩为概率
        self.model.add(Dense(self.dataset.num_classes))
        self.model.add(Activation('softmax'))  # softmax 压缩数据用来分类
        self.model.summary()

    def train_model(self):
        """
        训练模型, 添加参数
        进行模型训练的函数，具体的optimizer、loss可以进行不同选择
        :return:
        """
        # 损失函数和优化方式
        self.model.compile(
            optimizer='adam',  # adam, 带有动量的, 对学习率进行优化的 adaptive momentum
            loss='categorical_crossentropy',
            metrics=['accuracy'])

        # epochs、batch_size为可调的参数，epochs为训练多少轮、batch_size为每次训练多少个样本
        self.model.fit(self.dataset.X_train, self.dataset.Y_train, epochs=25, batch_size=5)

    def evaluate_model(self):
        print('\nTesting---------------')
        loss, accuracy = self.model.evaluate(self.dataset.X_test, self.dataset.Y_test)

        print('test loss;', loss)
        print('test accuracy:', accuracy)

    def save(self, file_path=FILE_PATH):
        print('Model Saved.')
        self.model.save(file_path)

    def load(self, file_path=FILE_PATH):
        print('Model Loaded.')
        self.model = load_model(file_path)

    def predict(self, img):
        """
        需要确保输入的img得是灰化之后（channel =1 )且 大小为IMAGE_SIZE的人脸图片
        :param img:
        :return:
        """
        img = img.reshape((1, self.IMAGE_SIZE, self.IMAGE_SIZE,1))
        img = img.astype('float32')
        img = img/255.0

        result = self.model.predict_proba(img)  # 计算概率
        max_index = np.argmax(result)  # 选最佳选项

        return max_index, result[0][max_index]


if __name__ == '__main__':
    dataset = DataSet('face')
    model = Model()
    model.load_trainData(dataset)
    model.build_model()
    model.train_model()
    model.evaluate_model()
    model.save()














