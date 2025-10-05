from Stitcher import Stitcher
import cv2

# 读取拼接图片
imageA = cv2.imread("qq1.jpg")
imageB = cv2.imread("qq2.jpg")
imgec = cv2.imread("qq3.jpg")
# 把图片拼接成全景图
stitcher = Stitcher()
(result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
stitcher1 = Stitcher()
(result1, vis1) = stitcher1.stitch([result, imgec], showMatches=True)
# 显示所有图片
#cv2.imshow("Image A", imageA)
#cv2.imshow("Image B", imageB)
#cv2.imshow("Keypoint Matches", vis)
result1 = cv2.resize(result1, None, fx=0.3, fy=0.3)

cv2.imshow("Result", result1)
cv2.waitKey(0)
cv2.destroyAllWindows()