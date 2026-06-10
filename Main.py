import cv2 as cv
import numpy as np
import scipy as scip
import math
import matplotlib.pyplot as plt
from tangLine import findTangent

def dif(a, b):
    return np.abs(a[0]-b[0]) + np.abs(a[1]-b[1]) + np.abs(a[2]-b[2])

threshold = 100
firstDerSmplCount = 100
firstDerSmplSize = 30

image = cv.imread("Fluid.png")

assert image is not None, "Image not found"

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
       
curIndex = 0

regResults = []

for num in range(firstDerSmplCount):

    curIndex = int(np.round(num * (image.shape[1] - firstDerSmplSize) / (firstDerSmplCount - 1)))

    m, b = np.polyfit(np.arange(curIndex, curIndex + firstDerSmplSize), yCoords[curIndex:(curIndex + firstDerSmplSize)], 1)
    regResults.append(-m)
    print(num, -m)

    
    cv.line(image2, (curIndex + firstDerSmplSize // 2, 0), (curIndex + firstDerSmplSize // 2, image.shape[0]), (255, 0, 0), 1)
    cv.line(image2, (curIndex, yCoords[curIndex] - 15), (curIndex + firstDerSmplSize, findTangent((curIndex, yCoords[curIndex] - 15), firstDerSmplSize, regResults[len(regResults)-1], image.shape)), (0, 255, 0), 1)


regResAbs = list(map(np.abs,regResults))
regResAbs = [round(math.atan(x) * 180 / np.pi, 3) for x in regResAbs]

mean = np.mean(regResAbs)
sigma = np.std(regResAbs)


plt.hist(regResAbs)
plt.title("Histogram of slopes")
plt.text(3,2,"Mean: "+str(round(mean,3)))
plt.text(10,2,"SD: "+str(round(sigma,3)))

#cv.imshow("image", image)
cv.imshow("image 2", image2)

plt.show()
