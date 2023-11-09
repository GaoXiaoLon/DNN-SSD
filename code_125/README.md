## SSD实现目标检测

✔️ OpenCV的DNN模块支持常见得对象检测模型SSD， 以及Mobile Net-SSD。

✔️ 这里我们用基于Caffe训练好的mobile-net SSD来测试目标检测。


### 视频检测

```python
import cv2

# 定义模型文件的路径
model_bin = "D:/Desktop/Python/model/ssd/MobileNetSSD_deploy.caffemodel";
config_text = "D:/Desktop/Python/model/ssd/MobileNetSSD_deploy.prototxt";

objName = ["background",    # 背景
"aeroplane", "bicycle", "bird", "boat", # 飞机、自行车、鸟、船
"bottle", "bus", "car", "cat", "chair", # 瓶子、公共汽车、汽车、猫、椅子
"cow", "diningtable", "dog", "horse",   # 牛、餐桌、狗、马
"motorbike", "person", "pottedplant",   # 摩托车、人、盆栽植物
"sheep", "sofa", "train", "tvmonitor"]; # 绵羊、沙发、火车、电视监视器

# 加载caffe模型
net = cv2.dnn.readNetFromCaffe(config_text, model_bin)

# 打开摄像头
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if ret is False:
        break
    h, w = frame.shape[:2]

    # 生成图像的blob
    blobImage = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), (127.5, 127.5, 127.5), True, False);
    net.setInput(blobImage)
    cvOut = net.forward()

    # 处理检测结果
    for detection in cvOut[0,0,:,:]:
        score = float(detection[2])
        objIndex = int(detection[1])
        if score > 0.5:
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h

            # 绘制边界框和标签
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
            cv2.putText(frame, "score:%.2f, %s"%(score, objName[objIndex]),
                    (int(left) - 10, int(top) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, 8);
    # 显示处理后的图像
    cv2.imshow('video-ssd-demo', frame)
    c = cv2.waitKey(10)
    if c == 27: # 按下ESC键退出循环
        break

# 释放摄像头资源
cv2.waitKey(0)
# 关闭所有OpenCV窗口
cv2.destroyAllWindows()
```

