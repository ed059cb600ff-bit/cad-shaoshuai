import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import myutils
#from images.识别图形颜色 import contours

#输入图像
img=cv2.imread('ocr_a_reference.png',1)
#灰度图
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#二值化
ret,thresh=cv2.threshold(gray_img,127,255,cv2.THRESH_BINARY_INV)
# 轮廓检测
contours,hierarchy=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

refCnts=myutils.sort_contours(contours,method='left-to-right')[0]
digits={}

for (i,c) in enumerate(refCnts):
    (x,y,w,h)=cv2.boundingRect(c)
    roi=gray_img[y:y+h,x:x+w]
    roi=cv2.resize(roi,(57,88))
    retxx,roi=cv2.threshold(roi,127,255,cv2.THRESH_BINARY_INV)
    digits[i]=roi


rectkernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqkernel=cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
cv2.drawContours(img,contours,-1,(0,255,0),3)

image1=cv2.imread('credit_card_01.png',1)
image=image1.copy()

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#二值化
ret1,thresh1=cv2.threshold(gray_image,127,255,cv2.THRESH_BINARY)
tophat=cv2.morphologyEx(thresh1,cv2.MORPH_TOPHAT,rectkernel)


#膨胀
gradx=cv2.dilate(thresh1,None,iterations=9)
#腐蚀
gradx=cv2.erode(gradx,None,iterations=9)
thresh=cv2.threshold(gradx,0,255,
    cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]
#轮廓
contours1,hierarchy1=cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image.copy(),contours1,-1,(0,255,0),3)
locs=[]
for (i,c) in enumerate(contours1):
    (x,y,w,h)=cv2.boundingRect(c)
    ar=1.0*w/h
    if 3.2<=ar<=3.5:
        if(50<=w and 20<=h<50):
            locs.append((x,y,w,h))
locs=sorted(locs,key=lambda x:x[0])
output=[]
for (i,(gX,gY,gW,gH)) in enumerate(locs):
    groupOutput: list[str]=[]
    group=gray_image[gY-5:gY+gH+5,gX-5:gX+gW+5]
    group=cv2.threshold(group,0,255
        ,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

    digits1,qqq=cv2.findContours(group.copy(),
         cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    digitss=myutils.sort_contours(digits1,
            method='left-to-right')[0]
    for c in digitss:
        (x,y,w,h)=cv2.boundingRect(c)
        roi=group[y:y+h,x:x+w]
        roi=cv2.resize(roi,(57,88))
       # cv2.imshow('img', roi)
      #  cv2.waitKey(0)
        socres=[]

        for (digit, digitROI) in digits.items():
            result=cv2.matchTemplate(roi,digitROI,cv2.TM_CCOEFF)
            (_,score,maxVal,maxLoc)=cv2.minMaxLoc(result)
            socres.append(score)
        groupOutput.append(str(np.argmax(socres)))

    cv2.rectangle(image,(gX-5,gY-5),(gX+gW+5,gY+gH+5),(0,0,255),1)
    cv2.putText(image,"".join(groupOutput),(gX,gY-15),
    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,0,255),2)
    output.extend('123')
print("Credit Card Type: {}".format("".join(output)))
cv2.imshow('img',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

