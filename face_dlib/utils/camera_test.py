# OpenCv 调用摄像头
# 默认调用笔记本摄像头


import cv2

cap = cv2.VideoCapture(0)

# cap.set(propId, value)
# 设置视频参数: propId - 设置的视频参数, value - 设置的参数值
# cap.set(3, 480) cap.set(4, 360) 调整宽和高
cap.set(3, 480)

# cap.isOpened() 返回 true/false, 检查摄像头初始化是否成功
print(cap.isOpened())

# cap.read()
"""
返回两个值
    先返回一个布尔值, 如果视频读取正确, 则为 True, 如果错误, 则为 False; 
    也可用来判断是否到视频末尾;
    
    再返回一个值, 为每一帧的图像, 该值是一个三维矩阵;
    
    通用接收方法为: 
        ret,frame = cap.read();
        ret: 布尔值;
        frame: 图像的三维矩阵;
        这样 ret 存储布尔值, frame 存储图像;
"""

# 不断刷新cv2画板, 形成视频
while cap.isOpened():
    ret, img = cap.read()
    # 每帧数据延时 1ms, 如果延时为0, 读取的是静态帧
    cv2.imshow("camera", img)
    k = cv2.waitKey(1)

    # 按下 's' 保存截图
    if k == ord('s'):
        cv2.imwrite("test.jpg", img)

    # 按下 'q' 退出
    if k == ord('q'):
        break

# 释放摄像头
cap.release()

# 关闭窗口
cv2.destroyAllWindows()