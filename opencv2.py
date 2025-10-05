import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('321.jpg',1)
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

blur=cv2.GaussianBlur(img,(5,5),1)#模糊
ret,thresh=cv2.threshold(gray_img,200,255,cv2.THRESH_BINARY_INV)

kernel=np.ones((5,5),np.uint8)

erosion=cv2.erode(thresh,kernel,iterations=1)   #腐蚀
dilat=cv2.dilate(thresh,kernel,iterations=1)   #膨胀
open=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)#开运算
close=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)#闭运算
gradient=cv2.morphologyEx(dilat,cv2.MORPH_GRADIENT,kernel)#梯度






   #soble
soblex=cv2.Sobel(close,cv2.CV_64F,0,1,ksize=5)#x轴
sobley=cv2.Sobel(close,cv2.CV_64F,1,0,ksize=5)#y轴
soblex = cv2.convertScaleAbs(soblex)
sobley = cv2.convertScaleAbs(sobley)
scole=cv2.addWeighted(soblex,1,sobley,1,0)#加权


canny=cv2.Canny(blur,200,256)
imgg=cv2.imread('img.png',0)

contours,binay=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)#轮廓
imgcopy=img.copy()
res=cv2.drawContours(imgcopy,contours,-1,(0,0,255),2)#画出轮廓
cv2.imshow('image',res)
cv2.waitKey(0)
cv2.destroyAllWindows()


