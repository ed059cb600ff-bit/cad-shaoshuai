import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import myutils

lll=cv2.imread('img_2.png',0)
#e二值化
ret,th1 = cv2.threshold(lll,110,255,cv2.THRESH_BINARY)
cv2.imwrite('img_2_1.png',th1)

