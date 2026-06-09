import cv2 as cv
import numpy as np
import scipy as scip
import math

def dif(a, b):
    return np.abs(a[0]-b[0]) + np.abs(a[1]-b[1]) + np.abs(a[2]-b[2])



threshold = 100
numReg = 20

image = cv.imread("C:\\Users\\torpe\\Dropbox\\PC\\Downloads\\Fluid.png")

image2 = np.zeros(image.shape)

#image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
#image = image[::,::,1]

bgSample = [0,0,0]
for i in range(0,10):
    for j in range(0,10):
           bgSample[0] += int(image[2*i][2*j][0])
           bgSample[1] += int(image[2*i][2*j][1])
           bgSample[2] += int(image[2*i][2*j][2])

bgSample[0] //= 100
bgSample[1] //= 100
bgSample[2] //= 100

bgSample = np.array(bgSample)
yCoords = []

for y in range(image.shape[1]):
    found = False
    for x in range(image.shape[0]):
        pixel = image[x, y]
        if dif(pixel, bgSample) > threshold:
            image2[x,y] = np.array([255, 255, 255])
            yCoords.append(x)
            found = True
            break

    if not found:
        yCoords.append(image.shape[0])
       
regLen = int(np.round(len(yCoords)/numReg))
curIndex = 0

regResults = []

for num in range((regLen)):
   
    m,b = np.polyfit(np.array(range(curIndex,curIndex+len(yCoords[curIndex:curIndex+(num+1)*regLen]))),yCoords[curIndex:curIndex+(num+1)*regLen],1)
    print(-math.atan(m)*180/math.pi)       

cv.imshow("image", image)
cv.imshow("image 2", image2)

cv.waitKey(0)
