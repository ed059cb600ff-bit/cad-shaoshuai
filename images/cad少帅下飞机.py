import cv2
import numpy as np
from collections import defaultdict

def add(l,cv):
    startx=200
    starty=500
    mask=cv[startx:startx+720,starty:starty+1078,:]
    mask[l>100]=0
    cv[startx:startx+720,starty:starty+1078,:]=cv[startx:startx+720,starty:starty+1078,:]+l
    return cv

fourcc = cv2.VideoWriter_fourcc(*'mp4v')#视频保存格式
video = cv2.VideoWriter("output_video.mp4", fourcc, 30, (1919, 1079)) #保存文件名，视频格式，帧率，分辨率

cv=cv2.imread('img.png')#读取图片
cap = cv2.VideoCapture("sssfj.mp4", 0)#读取视频
##抽帧
if cap.isOpened():
    ret, frame = cap.read()
else:
    ret = False
while ret:
    ret, frame = cap.read()
    if ret == False:
        break
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)#灰度化
    ret2, thresh = cv2.threshold(gray_frame, 25, 255, cv2.THRESH_BINARY_INV)#二值化
    contour, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)#轮廓提取
    lm = np.zeros((720, 1078, 3))
    cv2.drawContours(lm, contour, -1, (255,255, 255), 1)    #画轮廓
    end=add(lm,cv.copy())   #cad界面融合少帅
    #保存视频
    video.write(end)#保存视频
    cv2.imshow('frame',end)#展示效果
    c = cv2.waitKey(1)
    if c == 27:  # ESC键退出
        break

cap.release()
video.release()

