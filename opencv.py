import cv2
import matplotlib.pyplot as plt
import numpy as np
img=cv2.imread('321.jpg',0)
img5=cv2.imread('321.jpg',1)
# b,g,r=cv2.split(img+126 )
img2=cv2.blur(img,(5,5))

ret,thresh=cv2.threshold(img,200,255,cv2.THRESH_BINARY)
#cv2.imshow('image',thresh)       #灰度二分化

kernel=np.ones((10,10),np.uint8)
erosion=cv2.erode(thresh,kernel,iterations=1)   #腐蚀
#cv2.imshow('image',erosion)

dilat=cv2.dilate(erosion,kernel,iterations=1)   #膨胀
cv2.imshow('image',dilat)
ret,ll=cv2.threshold(dilat,200,255,cv2.THRESH_BINARY_INV)
#cv2.imshow('image',ll)

#开运算，先膨胀，后腐蚀
opening=cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)
#闭运算，先腐蚀，后膨胀
closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernel)
#h=np.hstack([thresh,opening,closing])

#cv2.imshow('image',h)
#梯度
gradient=cv2.morphologyEx(erosion,cv2.MORPH_GRADIENT,kernel)
#cv2.imshow('image',gradient)

# 创建一个3通道的彩色图像
height, width = img.shape
img3 = np.zeros((height, width, 3), dtype=np.uint8)
img3[:,:,2] = gradient  # 红色通道设为梯度图像

# 将灰度图像转换为3通道图像
gradient_3channel = cv2.cvtColor(gradient, cv2.COLOR_GRAY2BGR)
ll_3channel = cv2.cvtColor(ll, cv2.COLOR_GRAY2BGR)

# 现在所有图像都是3通道的，可以安全地进行水平堆叠
result = np.hstack([img5,img3, gradient_3channel, ll_3channel])
cv2.imshow('Combined Result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
