import dlib
import numpy as np
import PIL.Image


# Dlib 人脸检测器和预测器
pwd = "/Users/zehua/GoogleDrive/PythonProject/Face_Recognition/face_dlib/"
PREDICT_MODEL_68_PATH = pwd+'utils/models/shape_predictor_68_face_landmarks.dat'
# 用于识别人脸位置
face_detector = dlib.get_frontal_face_detector()
# 通过传入人脸图片, 获取人脸的68个特征点
pose_predictor_68_point = dlib.shape_predictor(PREDICT_MODEL_68_PATH)

# 人脸识别模型，用于提取128D的特征矢量
MODEL_RESNET_PATH= pwd+"utils/models/dlib_face_recognition_resnet_model_v1.dat"
facerec = dlib.face_recognition_model_v1(MODEL_RESNET_PATH)


def _rect_to_css(rect):
    """
    将dlib的rect对象, 转为(上, 右, 下, 左)的顺序
    :param rect: dlib的rect对象
    :return: (上, 右, 下, 左)
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    将(上, 右, 下, 左)的tuple, 转为与dlib的rect的一致顺序(左, 上, 右, 下)
    :param css:
    :return:
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_to_bounds(origin, image_shape):
    """
    在屏幕范围内裁剪图片, 返回位置
    :param css: 子图
    :param image_shape: image的大小
    :return: 处理后的子图范围
    """
    return max(origin[0], 0), min(origin[1], image_shape[1]), min(origin[2], image_shape[0]), max(origin[3], 0)


def _raw_face_locations(img, number_of_times_to_upsample=1):
    """
    识别图片中的所有人脸
    :param img: 包含或不包含人脸的未知图片
    :param number_of_times_to_upsample:
    :return: 返回识别出人脸的位置
    """
    return face_detector(img, number_of_times_to_upsample)


def get_face_locations(img, number_of_times_to_upsample=1):
    """
    (上, 右, 下, 左)顺序
    获取一张图片中所有人脸的位置, 中间进行了一项操作:将人脸的bounding位置都限定在图内
    :param img: 包含或不包含人脸的位置图片
    :param number_of_times_to_upsample: 上采样值, 更大的值能找到更小的脸
    :return: 一个列表, 包含所有人脸的位置
    """

    return [_trim_to_bounds(_rect_to_css(face), img.shape) for face in face_detector(img, number_of_times_to_upsample)]


def get_face_encodings(face_image, face_locations=None):
    """
    在face_image中(指定位置)识别人脸, 并根据pose_predictor_68获取68个特征点, 然后根据face_descriptor训练这68个点, 输出[1, 128]的特征值
    :param face_image: 人脸或包含人脸的图片
    :param face_locations: 如果给定位置, 就只识别该位置的脸
    :return: 返回一个列表, 包含了所有人脸的128D特征
    """
    face_image = face_image[:, :, ::-1]

    if face_locations:
        face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    elif face_locations is None:
        face_locations = _raw_face_locations(face_image)

    landmarks = get_68_points(face_image, face_locations)
    return [np.array(facerec.compute_face_descriptor(face_image, landmark)) for landmark in landmarks]


def get_68_points(face_image, face_locations=None):

    # 或取68个面部关键点
    landmarks = [pose_predictor_68_point(face_image, face_location) for face_location in face_locations]
    return landmarks


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
