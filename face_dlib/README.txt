文件目录:
|get_face.py        按s保存人脸截图到src/中
|train.py           提取src/中所有人脸的128特征值, 储存到data/features_all.txt中
|face_reco.py       打开摄像头实时检测人脸, 并在标记姓名和id
|utils/
  ├──camera_test.py  打开以测试能否调用摄像头
  ├──db_utils.py     用于数据存储的模块
  ├──face_utils.py   用于处理人脸信息的模块
  ├──models/
    ├──shape_predictor_68_face_landmarks.dat      人脸landmark检测模型
    ├──dlib_face_recognition_resnet_model_v1.dat  人脸128特征提取模型
    ├──shape_predictor_5_face_landmarks.data      人脸landmark检测模型
|data/
  ├──features_all.txt 每一行一条人脸数据, 类型为dict(json), 例如:{'_id':242, 'name':'zhou', 'encoding':[0.121321412, -0.0212323, ....]}
  ├──useless/         该文件夹存储了一些工程测试用代码, 用不到
|src/              储存人脸图片文件夹, 文件名代表name
  ├──hou.jpg
  ├──lam.jpg
  ├──1.jpg
  ├──...


执行流程:
1. get_face.py     按S截取人脸, 自动保存到src/
2. train.py        将src/中的人脸提取embedding到features_all.txt中
3. face_reco.py    先加载features_all.txt中的数据, 实时识别人脸

备注: embedding, encoding, 特征值, 128D特征值都是同一个意思, 专业术语应为embedding特征值
