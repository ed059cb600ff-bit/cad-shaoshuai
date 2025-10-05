import cv2
import numpy as np
from collections import defaultdict
cap = cv2.VideoCapture(0)
ret,img = cap.read()
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    detector = cv2.QRCodeDetector()
    data, bbox, _ = detector.detectAndDecode(img)
    if bbox is not None:
        print("解码数据:", data)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blurred = cv2.GaussianBlur(inverted, (5, 5), 200)
        inverted_blur = 255 - blurred
        sketch = cv2.divide(gray, inverted_blur, scale=256.0)
        # 绘制检测框
        cv2.polylines(sketch , [bbox.astype(int)], True, (0, 255, 0), 2)

        cv2.imshow('frame',sketch )
        if cv2.waitKey(1) & 0xFF == ord('q'):
           break
cap.release()
cv2.destroyAllWindows()