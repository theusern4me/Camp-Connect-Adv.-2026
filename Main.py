import cv2 as cv
from networkx import second_order_centrality
import numpy as np
import scipy as scip
import math
import matplotlib.pyplot as plt

def rnd(a):
    return int(np.round(a))

def findTangent(home, end, slope, imageSize):
    imageX , imageY = imageSize[0], imageSize[1] # imageSize is an array of two .shape values (x,y)
    homeX , homeY = home[0], home[1] # home is an array of two values (x,y) to dictate where the starting point is
    # end is one value (x) to dictate where the ending vertical is
    endY = homeY - (slope * end)
    return int(np.round(endY))

threshold = 100
firstDerThreshold = 0.5
secondDerThreshold = 0.00375
firstDerSmplCount = 100
firstDerSmplSize = 30
windowLength = 60
polyOrder = 2

image = cv.imread("Fluid.png")
assert image is not None, "Image not found"
image2 = np.zeros(image.shape)

bgSample = np.zeros(3)
for i in range(0,10):
    for j in range(0,10):
        bgSample += (image[2*i][2*j]).astype(np.uint8)

bgSample //= 100

yCoords = []
for y in range(image.shape[1]):
    found = False
    for x in range(image.shape[0]):
        if sum(map(np.abs, (image[x, y] - bgSample))) > threshold:
            
            yCoords.append(x)
            found = True
            break

    if not found: 
        yCoords.append(image.shape[0])
 
yCoords = scip.signal.savgol_filter(yCoords, window_length = windowLength, polyorder = polyOrder)

for y in range(image.shape[1]):
    image2[rnd(yCoords[y]),y] = np.array([255, 255, 255])

curIndex = 0

firstDerivative = []

for i in range(firstDerSmplCount):
    curIndex = int(np.round(i * (image.shape[1] - firstDerSmplSize) / (firstDerSmplCount - 1)))

    m, b = np.polyfit(np.arange(curIndex, curIndex + firstDerSmplSize), yCoords[curIndex:(curIndex + firstDerSmplSize)], 1)
    firstDerivative.append((-m, curIndex + firstDerSmplSize // 2))

    cv.circle(image2, (firstDerivative[i][1], int(image.shape[0]//2 - firstDerivative[i][0] * 200)), 2, (0, 255, 255), thickness = -1)

secondDerivative = []

filteredSlopes = []
filteredSlopes2 = []

for i in range(firstDerSmplCount - 2):
    secondDerivative.append(((firstDerivative[i + 2][0] - firstDerivative[i][0]) / (firstDerivative[i + 2][1] - firstDerivative[i][1]), firstDerivative[i + 1][1]))
    cv.circle(image2, (secondDerivative[i][1], int(image.shape[0]//2 - secondDerivative[i][0] * 3000)), 2, (0, 0, 255), thickness = -1)

    if np.abs(secondDerivative[i][0]) < secondDerThreshold:
        curIndex = int(np.round(firstDerivative[i + 1][1] - firstDerSmplSize // 2))
        filteredSlopes.append((firstDerivative[i + 1][0], i + 1))

firstDerMax = max([firstDerivative[x][0] for x in range(firstDerSmplCount)])

cv.line(image2, (0, image.shape[0] // 2 + rnd(firstDerMax * firstDerThreshold * 200)), (image.shape[1], image.shape[0] // 2 + rnd(firstDerMax * firstDerThreshold * 200)), (0, 64, 64), 1)
cv.line(image2, (0, image.shape[0] // 2 - rnd(firstDerMax * firstDerThreshold * 200)), (image.shape[1], image.shape[0] // 2 - rnd(firstDerMax * firstDerThreshold * 200)), (0, 64, 64), 1)
cv.line(image2, (0, image.shape[0] // 2 + rnd(secondDerThreshold * 3000)), (image.shape[1], image.shape[0] // 2 + rnd(secondDerThreshold * 3000)), (0, 0, 64), 1)
cv.line(image2, (0, image.shape[0] // 2 - rnd(secondDerThreshold * 3000)), (image.shape[1], image.shape[0] // 2 - rnd(secondDerThreshold * 3000)), (0, 0, 64), 1)
cv.line(image2, (0, image.shape[0] // 2 + rnd(firstDerMax * 200)), (image.shape[1], image.shape[0] // 2 + rnd(firstDerMax * 200)), (0, 64, 64), 1)
cv.line(image2, (0, image.shape[0] // 2 - rnd(firstDerMax * 200)), (image.shape[1], image.shape[0] // 2 - rnd(firstDerMax * 200)), (0, 64, 64), 1)

for i in range(len(filteredSlopes)):
    if np.abs(firstDerivative[filteredSlopes[i][1]][0]) > firstDerThreshold * firstDerMax:
        curIndex = int(np.round(firstDerivative[filteredSlopes[i][1]][1] - firstDerSmplSize // 2))
        cv.line(image2, (curIndex, rnd(yCoords[curIndex]) - 15), (curIndex + firstDerSmplSize, findTangent((curIndex, rnd(yCoords[curIndex]) - 15), firstDerSmplSize, firstDerivative[filteredSlopes[i][1]][0], image.shape)), (0, 255, 0), 1)
        filteredSlopes2.append(firstDerivative[filteredSlopes[i][1]][0])

regResAbs = list(map(np.abs, filteredSlopes2))
regResAbs = [round(math.atan(x) * 180 / np.pi, 3) for x in regResAbs]

mean = np.mean(regResAbs)
sigma = np.std(regResAbs)

plt.hist(regResAbs)
plt.title("Histogram of slopes")
plt.text(3,2,"Mean: "+str(round(mean,3)))
plt.text(2,2,"SD: "+str(round(sigma,3)))

print("Max Angle:", round(np.abs(max(regResAbs)), 2))
print("Mean: "+str(round(mean,2)))
print("SD: "+str(round(sigma,2)))
print("Coefficient of variation: "+ str(round(sigma/mean,2)))

cv.imshow("image 2", image2)

plt.show()
