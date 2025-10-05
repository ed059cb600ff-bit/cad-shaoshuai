import cv2
import numpy as np
from collections import defaultdict
cap = cv2.VideoCapture(0)
ret,img = cap.read()
detector = cv2.QRCodeDetector()

data, bbox, _ = detector.detectAndDecode(img)
while True:
    ret, img = cap.read()
    if not ret:
        break
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    if bbox is not None:
        print("解码数据:", data)
        # 绘制检测框
        cv2.polylines(img, [bbox.astype(int)], True, (0, 255, 0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()