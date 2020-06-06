import cv2
from utils.face_utils import get_face_locations

"""
全局变量
"""
FONT = cv2.FONT_HERSHEY_DUPLEX
FACES_PATH = "src/"


# 建cv2摄像头对象，这里使用电脑自带摄像头，如果接了外部摄像头，则自动切换到外部摄像头
# 设置视频参数，propId设置的视频参数，value设置的参数值
cap = cv2.VideoCapture(0)
cap.set(3, 480)

# 记录已经保存的人脸
count = 0
while cap.isOpened():
    flag, img = cap.read()

    """
    变量预定义
    """
    kk = cv2.waitKey(1)
    # 图像的高和宽
    h, w = img.shape[0], img.shape[1]
    # 用于储存识别出的人脸, 每帧开始时清空
    face_img_list = []


    """
    人脸检测及图像预处理模块
    """
    small_frame = img
    # 缩小1/4, 加快检测速度
    small_frame = cv2.resize(small_frame, (0, 0), fx=0.25, fy=0.25)
    face_locations = get_face_locations(small_frame)
    # face坐标顺序(上, 右, 下, 左)
    for face in face_locations:
        # top, right, bottom, left = face
        top, right, bottom, left = face[0]* 4,face[1]* 4,face[2]* 4,face[3]* 4

        # 截图扩大1/3, 便于保存后训练人脸特征, 扩大的原因是, 再次用dlib识别人脸时, 避免原图片只有人脸部分
        height, width = bottom-top, right-left
        top_large, bottom_large = top - height//3, bottom + height//3
        left_large, right_large = left-width//3, right + width//3

        # 判断头像子图是否完整(是否超出整张图)
        if (top_large < 1) or (h - bottom_large < 1) or (left_large < 1) or (w - right_large < 1):
            cv2.rectangle(img, (left_large, top_large), (right_large, bottom_large), (0, 0, 255), 2)

        else:
            cv2.rectangle(img, (left_large, top_large), (right_large, bottom_large), (255, 255, 255), 2)

            # 区域截图, image[上:下,左:右]
            im_blank = img[top_large:bottom_large, left_large:right_large]
            face_img_list.append(im_blank)

        # 标出人脸数
        cv2.putText(img, "Faces: " + str(len(face_locations)), (20, 50), FONT, 1, (0, 0, 255), 1, cv2.LINE_AA)

    if len(face_locations) == 0:
        # 没有检测到人脸
        cv2.putText(img, "No Face", (20, 50), FONT, 1, (0, 0, 255), 1, cv2.LINE_AA)

    """
    按键触发模块(基于每帧)
    """
    if kk == ord('s'):
        _id = 0
        for index in range(len(face_img_list)):
            count += 1
            cv2.imwrite(FACES_PATH + "face" + str(count) + ".jpg", face_img_list[index])
            print("写入本地 / Save into：", str(index) + "/img_face_" + str(count) + ".jpg")

    # 按下q键退出
    if kk == ord('q'):
        break

    cv2.putText(img, "Press S to Save", (20, 200), FONT, 1, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow('Face Recognition', img)


# 释放摄像头
cap.release()
# 删除建立的窗口
cv2.destroyAllWindows()