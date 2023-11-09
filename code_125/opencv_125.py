# 导入OpenCV库
import cv2

# 定义了模型文件路径model_bin和配置文件路径config_text，这些文件是用于加载的Caffe模型。
model_bin = "D:/Desktop/Python/model/ssd/MobileNetSSD_deploy.caffemodel";
config_text = "D:/Desktop/Python/model/ssd/MobileNetSSD_deploy.prototxt";

# objName是一个列表，包含了所有可能的物体类别名称
objName = ["background",    # 背景
"aeroplane", "bicycle", "bird", "boat", # 飞机、自行车、鸟、船
"bottle", "bus", "car", "cat", "chair", # 瓶子、公共汽车、汽车、猫、椅子
"cow", "diningtable", "dog", "horse",   # 牛、餐桌、狗、马
"motorbike", "person", "pottedplant",   # 摩托车、人、盆栽植物
"sheep", "sofa", "train", "tvmonitor"]; # 绵羊、沙发、火车、电视监视器

# 加载Caffe模型
net = cv2.dnn.readNetFromCaffe(config_text, model_bin)

# 打开摄像头
cap = cv2.VideoCapture(0)

# 打开摄像头进行视频捕获。在无限循环中，它读取摄像头的帧，如果读取失败则跳出循环
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    h, w = frame.shape[:2]

    # 使用cv2.dnn.blobFromImage将帧转换成一个blob图像（一种用于神经网络的图像格式）
    blobImage = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False);

    # 将blob图像作为输入传递给神经网络
    net.setInput(blobImage)

    # 获取网络的输出
    cvOut = net.forward()

    # 对于网络的每个输出，它提取出得分（第二个元素）和对象索引（第三个元素）
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        objIndex = int(detection[1])

        # 如果得分大于0.5，它将在帧上绘制一个矩形框，表示检测到的对象的位置
        if score > 0.5:
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h

            # 绘制边界框和标签，在框上添加一个文本，显示对象的得分和类别
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
            cv2.putText(frame, "score:%.2f, %s"%(score, objName[objIndex]),
                    (int(left) - 10, int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8);

    # 显示处理后的图像
    cv2.imshow('video-ssd-demo', frame)

    # 如果用户按下ESC键（键码为27），则跳出循环
    c = cv2.waitKey(10)
    if c == 27:
        break

# 释放摄像头资源
cv2.waitKey(0)
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()