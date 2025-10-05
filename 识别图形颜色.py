import cv2
import numpy as np
from collections import defaultdict

# 定义颜色范围（HSV空间）
color_ranges = {
    "red": ([0, 100, 100], [10, 255, 255]),
    "red2": ([160, 100, 100], [179, 255, 255]),
    "green": ([40, 50, 50], [80, 255, 255]),
    "blue": ([100, 50, 50], [130, 255, 255]),
    "yellow": ([20, 100, 100], [40, 255, 255]),
    "purple": ([130, 50, 50], [160, 255, 255]),
    "orange": ([10, 100, 100], [20, 255, 255])
}


# 形状检测函数
def detect_shape(c):
    shape = "unknown"
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    # 根据顶点数判断形状
    if len(approx) == 3:
        shape = "triangle"
    elif len(approx) == 4:
        # 检查是否为矩形
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h
        if 0.9 <= aspect_ratio <= 1.1:
            shape = "square"
        else:
            shape = "rectangle"
    elif len(approx) >= 8:
        # 圆形检测
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            circularity = 4 * np.pi * area / (hull_area * hull_area)
            if circularity > 0.8:
                shape = "circle"
    return shape


# 颜色识别函数
def detect_color(hsv_img, contour):
    mask = np.zeros(hsv_img.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    mean_val = cv2.mean(hsv_img, mask=mask)[:3]

    color_name = "unknown"
    min_dist = float("inf")

    for color, (lower, upper) in color_ranges.items():
        # 计算HSV颜色空间中的欧氏距离
        range_center = np.array([(lower[0] + upper[0]) / 2,
                                 (lower[1] + upper[1]) / 2,
                                 (lower[2] + upper[2]) / 2])
        dist = np.linalg.norm(np.array(mean_val) - range_center)

        if dist < min_dist:
            min_dist = dist
            color_name = color

    # 合并红色范围
    if color_name == "red2":
        color_name = "red"

    return color_name


# 主程序
cap = cv2.VideoCapture(0)

#角点检测函数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#LUCAS-KANADE光流法参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

#第一帧
ret, frame = cap.read()


while True:
    ret, frame = cap.read()
    if not ret:
        break
    # 图像预处理
    blurred = cv2.GaussianBlur(frame, (7, 7), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )

    # 形态学操作
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 按面积过滤轮廓
    valid_contours = []
    for c in contours:
        area = cv2.contourArea(c)
        if 500 < area < 50000:  # 过滤太大或太小的轮廓
            valid_contours.append(c)

    # 存储检测结果
    detected_shapes = defaultdict(list)

    # 处理每个有效轮廓
    for c in valid_contours:
        shape = detect_shape(c)
        if shape != "unknown":
            color_name = detect_color(hsv, c)

            # 计算中心点
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # 存储结果
                detected_shapes[(cX, cY)] = (shape, color_name)

                # 在中心显示结果
                text = f"{color_name} {shape}"
                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.putText(frame, text, (cX - text_size[0] // 2, cY),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 显示结果
    cv2.imshow("Shape Detection", frame)

    # 退出条件
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()