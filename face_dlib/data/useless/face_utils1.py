import dlib
import numpy as np
import PIL.Image


# Dlib 人脸检测器和预测器
PREDICT_MODEL_68_PATH = 'utils/models/shape_predictor_68_face_landmarks.dat'
# 用于识别人脸位置
face_detector = dlib.get_frontal_face_detector()
# 通过传入人脸图片, 获取人脸的68个特征点
pose_predictor_68_point = dlib.shape_predictor(PREDICT_MODEL_68_PATH)

# 人脸识别模型，用于提取128D的特征矢量
MODEL_RESNET_PATH= "utils/models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(MODEL_RESNET_PATH)


def get_face_locations(img, number_of_times_to_upsample=1):
    """
    获取一张图片中所有人脸的位置, 中间进行了一项操作:将人脸的bounding位置都限定在图内
    :param img: 包含或不包含人脸的位置图片
    :param number_of_times_to_upsample:
    :return:
    """

    return face_detector(img, number_of_times_to_upsample)


def get_face_encodings(face_image, face_locations):
    """
    在face_image中(指定位置)识别人脸, 并根据pose_predictor_68获取68个特征点, 然后根据face_descriptor训练这68个点, 输出[1, 128]的特征值
    :param face_image: 人脸或包含人脸的图片
    :param face_locations: 只识别该位置的脸
    :return: 返回一个列表, 包含了所有人脸的128D特征
    """
    face_image = face_image[:, :, ::-1]
    if face_locations is None:
        face_locations = face_detector(face_image)

    # 或取68个面部关键点
    return np.array([facerec.compute_face_descriptor(face_image, pose_predictor_68_point(face_image, face_location)) for face_location in face_locations])


def face_distance(known_encodings, encoding):
    """
    计算encoding与known_encodings的Euclidean距离
    :param known_encodings: 已知的所有脸的encoding
    :param encoding: 给定的未知的encoding
    :return: 返回一个列表, 每个元素是与encoding的欧氏距离
    """
    if len(encoding) == 0:
        return np.empty((0))
    distance = np.linalg.norm(known_encodings - encoding, axis=1)
    return distance



def load_image_file(file, mode='RGB'):
    """
    读取图片
    :param file: 文件名
    :param mode:
    :return: 返回读取的图片
    """
    im = PIL.Image.open(file)
    if mode:
        im = im.convert(mode)
    return np.array(im)


if __name__ == '__main__':
    pass
