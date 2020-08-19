import cv2
import numpy as np
import os
import csv
from matplotlib import pyplot as plt

src = cv2.imread('img8.png', cv2.IMREAD_UNCHANGED)
print(src.shape)

    
#extract green channel
img = src[:,:,0]
#img = cv2.imread('k1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

Z = img.reshape((-1,3))
Z = np.float32(Z)
    
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)   
K=12
ret, label1, center1 = cv2.kmeans(Z, K, None,
                                      criteria, 12, cv2.KMEANS_PP_CENTERS)
center1 = np.uint8(center1)
res1 = center1[label1.flatten()]
output2 = res1.reshape((img.shape))

lab = cv2.cvtColor(output2, cv2.COLOR_BGR2LAB)

lab_planes = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=4.0,tileGridSize=(8, 8))

lab_planes[0] = clahe.apply(lab_planes[0])

lab = cv2.merge(lab_planes)

output2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

_, th1 = cv2.threshold(output2, 50, 255, cv2.THRESH_BINARY)

cv2.imshow('image', th1)
#cv2.imwrite('ex6.png', th1)#
cv2.waitKey(0)
cv2.destroyAllWindows()
