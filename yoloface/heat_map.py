from face_detector import YoloDetector
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------- 初始化模型和读取图片 --------------------
model = YoloDetector(weights_name='yolov5n_state_dict.pt',
                    config_name='yolov5n.yaml',
                    target_size=None,
                    device='cuda:0')

image_path = '/amax/tyut/user/lyh/lyh/AVCL/seq24-cam1/frame0210.jpg'
image = cv2.imread(image_path)
if image is None:
    print(f"Cannot load image: {image_path}")
    exit()

# -------------------- 检测人脸并获取结果 --------------------
bboxes, landmarks = model.predict(image)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
height, width = image.shape[:2]

# -------------------- 创建热图矩阵 --------------------
heatmap = np.zeros((height, width), dtype=np.float32)

# 定义高斯核参数
gaussian_size = 101  # 高斯核尺寸（奇数）
gaussian_std = 30    # 标准差（控制分布范围）
gaussian_kernel = cv2.getGaussianKernel(gaussian_size, gaussian_std)
gaussian_kernel = gaussian_kernel @ gaussian_kernel.T  # 生成二维高斯核
gaussian_kernel = gaussian_kernel / gaussian_kernel.max()  # 归一化到0-1

# -------------------- 根据检测框生成热图 --------------------
for bbox_group in bboxes:
    for bbox in bbox_group:
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        # 计算检测框中心
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        # 计算高斯核覆盖区域
        half_size = gaussian_size // 2
        y_start = max(0, center_y - half_size)
        y_end = min(height, center_y + half_size + 1)
        x_start = max(0, center_x - half_size)
        x_end = min(width, center_x + half_size + 1)
        # 截取高斯核有效部分
        kernel_y_start = max(half_size - center_y, 0)
        kernel_y_end = gaussian_size - max(center_y + half_size + 1 - height, 0)
        kernel_x_start = max(half_size - center_x, 0)
        kernel_x_end = gaussian_size - max(center_x + half_size + 1 - width, 0)
        # 叠加高斯核到热图
        heatmap[y_start:y_end, x_start:x_end] += gaussian_kernel[
            kernel_y_start:kernel_y_end,
            kernel_x_start:kernel_x_end
        ]

# -------------------- 归一化并转换为颜色热图 --------------------
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())  # 归一化到0-1
heatmap = (heatmap * 255).astype(np.uint8)  # 转换为0-255
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 应用颜色映射
heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)  # 转换为RGB

# -------------------- 叠加热图与原图 --------------------
alpha = 0.5  # 热图透明度
overlay = cv2.addWeighted(image_rgb, 1 - alpha, heatmap_color, alpha, 0)

# -------------------- 绘制检测框和关键点 --------------------
for bbox_group in bboxes:
    for bbox in bbox_group:
        if len(bbox) < 4:
            continue
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

for landmark_group in landmarks:
    for landmark in landmark_group:
        for (x, y) in landmark:
            cv2.circle(overlay, (int(x), int(y)), 2, (0, 255, 0), -1)

# -------------------- 保存并显示结果 --------------------
output_path = '/amax/tyut/user/lyh/lyh/AVCL/seq24-cam1/heat0210.jpg'
plt.imshow(overlay)
plt.axis('off')
plt.title('Detection Result with Heatmap')
plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
plt.show()

print(f"Image saved to: {output_path}")