from face_detector import YoloDetector
import cv2
import matplotlib.pyplot as plt

# 初始化模型
model = YoloDetector(weights_name='yolov5n_state_dict.pt', config_name='yolov5n.yaml', target_size=None, device='cuda:0')

# 读取图片
# image_path = '/amax/tyut/user/lyh/lyh/AV163/seq08-1p-0100/seq08-1p-0100_cam1_jpg/img/frame0200.jpg'
image_path = '/amax/tyut/user/lyh/lyh/AVCL/2.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Cannot load image: {image_path}")
    exit()

# 预测
bboxes, landmarks = model.predict(image)
print(bboxes)
print(landmarks)
# 检查是否有检测结果
if not bboxes:
    print(f"No faces detected in the image: {image_path}")
else:
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # # 绘制检测结果
    # for bbox in bboxes:
    #     if len(bbox) < 4:
    #         continue
    #     x1, y1, x2, y2 = bbox[:4]
    #     cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
    #
    # for landmark in landmarks:
    #     for (x, y) in landmark:
    #         cv2.circle(image_rgb, (int(x), int(y)), 2, (0, 255, 0), -1)

    # 绘制检测结果
    for bbox_group in bboxes:
        for bbox in bbox_group:
            if len(bbox) < 4:
                continue
            x1, y1, x2, y2 = bbox[:4]
            print(f"Drawing bbox: ({x1}, {y1}), ({x2}, {y2})")
            cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # 绘制关键点
    for landmark_group in landmarks:
        for landmark in landmark_group:
            for (x, y) in landmark:
                print(f"Drawing landmark: ({x}, {y})")
                cv2.circle(image_rgb, (int(x), int(y)), 2, (0, 255, 0), -1)

    # 保存图像
    output_path = '/amax/tyut/user/lyh/lyh/AVCL/output_image2.jpg'  # 指定保存的路径和文件名
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)  # 将RGB转回BGR，因为cv2使用BGR格式
    cv2.imwrite(output_path, image_bgr)  # 保存图像

    # 显示结果
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Face Detection Result')
    plt.show()

    print(f"Image saved to: {output_path}")
