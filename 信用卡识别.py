import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
#print("args",ap.parse_args())
ap.add_argument("-t", "--template", required=True,
	help="path to template OCR-A image")
#print("args",ap.parse_args())
ref=cv2.imread("ocr_a_reference.png",1)
gray_ref=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
threshold_ref=cv2.threshold(gray_ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

contours, hierarchy = cv2.findContours(threshold_ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(ref, contours, -1, (255, 0, 0), 3)
cv2.imshow("Image",ref)
cv2.waitKey(0)
cv2.destroyAllWindows()

