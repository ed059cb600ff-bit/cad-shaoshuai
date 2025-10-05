import cv2
import numpy as np
from collections import defaultdict



cap = cv2.VideoCapture(0)
#角点检测函数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

#LUCAS-KANADE光流法参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#随机颜色条
color=np.random.randint(0,255,(100,3))
ret, old_frame = cap.read()

old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #选择好的点
    good_new = p1[st==1]
    good_old = p0[st==1]
    #画出轨迹
    sa=0;
    sb=0;
    time=0;
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        sa=a+sa
        sb=b+sb
        mask = cv2.line(mask, (int(a),int(b)),(int(c),int(d)), color[i].tolist(), 2)
        frame = cv2.circle(frame.copy(),(int(a),int(b)),5,color[i].tolist(),-1)
        time=time+1;
    sa=int(sa/time)
    sb=int(sb/time)
    print(sa,sb)
    if 200<sa<400 and 200<sb<400:
       cv2.putText(frame, "win", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    #计算光流
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    #选择好的点
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    #更新上一帧的图像和特征点
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()