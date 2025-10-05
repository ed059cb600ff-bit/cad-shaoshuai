import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import myutils

def show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
img1=cv2.imread('qq1.jpg',1)
img2=cv2.imread('qq2.jpg',1)
img3=cv2.imread('qq3.jpg',1)
sift=cv2.xfeatures2d.SIFT_create()
kp1,des1=sift.detectAndCompute(img1,None)
kp2,des2=sift.detectAndCompute(img2,None)
#kp3,des3=sift.detectAndCompute(img3,None)
bf=cv2.BFMatcher(crossCheck=True)
matches=bf.match(des1,des2)
matches=sorted(matches,key=lambda x:x.distance)
img4=cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None,flags=2)
show('img3',img4)


