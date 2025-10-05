import cv2
import numpy as np

img = cv2.imread('123.png', 1)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 圆圈mask
lower_red1 = np.array([0, 70, 50])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 70, 50])
upper_red2 = np.array([180, 255, 255])
mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
lower_green = np.array([36, 70, 50])
upper_green = np.array([89, 255, 255])
mask_green = cv2.inRange(hsv, lower_green, upper_green)
lower_blue = np.array([90, 70, 50])
upper_blue = np.array([128, 255, 255])
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# 字母掩膜
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask_text = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((3,3), np.uint8)
mask_text = cv2.morphologyEx(mask_text, cv2.MORPH_OPEN, kernel, iterations=1)

# 去掉圆圈，只保留字母
mask_circles = cv2.bitwise_or(mask_red, mask_green)
mask_circles = cv2.bitwise_or(mask_circles, mask_blue)
mask_text_only = cv2.bitwise_and(mask_text, cv2.bitwise_not(mask_circles))

# 提取并变色、居中
def extract_and_recolor(mask, color_bgr):
    out = np.zeros_like(img)
    for i in range(3):
        out[:,:,i] = mask // 255 * color_bgr[i]
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return out
    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    crop = out[y_min:y_max+1, x_min:x_max+1]
    h_img, w_img = img.shape[:2]
    h_crop, w_crop = crop.shape[:2]
    base = np.zeros_like(img)
    y_start = (h_img - h_crop) // 2
    x_start = (w_img - w_crop) // 2
    base[y_start:y_start+h_crop, x_start:x_start+w_crop] = crop
    return base

purple = (180, 30, 180)
orange = (0, 140, 255)
yellow = (0, 255, 255)
white = (255, 255, 255)

img_red_new = extract_and_recolor(mask_red, purple)
img_green_new = extract_and_recolor(mask_green, orange)
img_blue_new = extract_and_recolor(mask_blue, yellow)
img_text_new = extract_and_recolor(mask_text_only, white)

h, w = img.shape[:2]
out = np.zeros((h*2, w*2, 3), dtype=np.uint8)
out[0:h, 0:w] = img_red_new
out[0:h, w:2*w] = img_green_new
out[h:2*h, 0:w] = img_blue_new
out[h:2*h, w:2*w] = img_text_new

cv2.imwrite('result_four_parts_letters_only.png', out)
