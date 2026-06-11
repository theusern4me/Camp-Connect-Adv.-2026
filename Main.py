import cv2 as cv
from networkx import second_order_centrality
import numpy as np
import scipy as scip
import math
import matplotlib.pyplot as plt

def rnd(a): return int(round(a)) # Rounds to an integer

backgroundThreshold = 80 # Tolerance for color being background
firstDerThreshold = 0.5 # Portion of maximum first derivitve needed to count as part of the slope
secondDerThreshold = 0.00375 # Maximum second derivative to count as part of the slope
thirdDerThreshold = 0.002 # Maximum third derivative to count as part of the slope
firstDerSmplCount = 1000 # Number of points derivative is calculated at
firstDerSmplSize = 30 # Sample size for each derivative
windowLength = 60 # Size of smoothing window
polyOrder = 2 # Order of smoothing polynomial
rThreshold = 0.7 # R value (Correlation Coeficcent) minimum value

image = cv.imread("gprtest2.JPG") # Gets image
assert image is not None, "Image not found" # Returns error if image is not found
image2 = np.zeros(image.shape) # Defines other windows
image3 = np.zeros(image.shape) 

bgSample = np.zeros(3) # Samples average background color
for i in range(100):
    bgSample += (image[2 * (i % 10)][2 * i // 10]).astype(np.uint8)

bgSample //= 100

edge = [] # Distance from the top of image to every pixel of the edge
for y in range(image.shape[1]):
    found = False
    for x in range(image.shape[0]): # Loops until a pixel is not part of the background 
        dif = int(sum(map(abs, (image[x, y] - bgSample))) // 3)        
        image2[x, y] = np.array([dif, dif, dif])
        
        if sum(map(abs, (image[x, y] - bgSample))) > backgroundThreshold and not found:
            edge.append(x)
            found = True

    if not found: 
        edge.append(image.shape[0])
 
smoothEdge = scip.signal.savgol_filter(edge, window_length = windowLength, polyorder = polyOrder) # Smooths edge

#for y in range(image.shape[1]): image2[rnd(smoothEdge[y]), y] = np.array([255, 255, 255]) # Draws edge
for y in range(image.shape[1]): image2[rnd(edge[y]), y] = np.array([255, 255, 255])

firstDerivative = [] # Derivatives of the edge
secondDerivative = []
thirdDerivative = []

for i in range(firstDerSmplCount): # Loops through and finds the first derivative at given number of points
    curIndex = rnd(i * (image.shape[1] - firstDerSmplSize) / (firstDerSmplCount - 1)) # Start of regression window
    x = np.arange(curIndex, curIndex + firstDerSmplSize)
    y = edge[curIndex:(curIndex + firstDerSmplSize)]

    rValue = np.corrcoef(x,y)[1,0] # Compute correlation strength for unsmoothed data
    
    m, b = np.polyfit(np.arange(curIndex, curIndex + firstDerSmplSize), smoothEdge[curIndex:(curIndex + firstDerSmplSize)], 1) # Performs regression to find the first deriviative
    firstDerivative.append((-m, curIndex + firstDerSmplSize // 2, abs(rValue)))

    cv.circle(image3, (firstDerivative[i][1], int(image.shape[0] // 4 - firstDerivative[i][0] * 150)), 2, (0, 255, 255), thickness = -1) # Draws circles to graph the first derivative

filteredSlopes = [] # Arrays for slopes that pass checks
filteredSlopes2 = []

for i in range(firstDerSmplCount - 2): # Finds second derivative and graphs
    secondDerivative.append(((firstDerivative[i + 2][0] - firstDerivative[i][0]) / (firstDerivative[i + 2][1] - firstDerivative[i][1]), firstDerivative[i + 1][1]))
    cv.circle(image3, (secondDerivative[i][1], int(image.shape[0] // 2 - secondDerivative[i][0] * 3000)), 2, (0, 0, 255), thickness = -1)

for i in range(firstDerSmplCount - 4): # Finds third derivative and graphs
    thirdDerivative.append(((secondDerivative[i + 2][0] - secondDerivative[i][0]) / (secondDerivative[i + 2][1] - secondDerivative[i][1]), secondDerivative[i + 1][1]))
    cv.circle(image3, (thirdDerivative[i][1], int(3 * image.shape[0] // 4 - thirdDerivative[i][0] * 20000)), 2, (255, 0, 255), thickness = -1)

    if abs(secondDerivative[i + 1][0]) < secondDerThreshold and abs(thirdDerivative[i][0]) < thirdDerThreshold and firstDerivative[i+2][2] > rThreshold: # Checks if second/third derivatives and R value are in bounds
        
        filteredSlopes.append((firstDerivative[i + 2][0], i + 1))

firstDerMax = max([abs(filteredSlopes[x][0]) for x in range(len(filteredSlopes))]) # Maximum first derivative that meets seconds and third derivative tests

# Plot cutoff lines
cv.line(image3, (0, image.shape[0] // 4 + rnd(firstDerMax * 150)), (image.shape[1], image.shape[0] // 4 + rnd(firstDerMax * 150)), (0, 64, 64), 1) # Draws bound lines
cv.line(image3, (0, image.shape[0] // 4 - rnd(firstDerMax * 150)), (image.shape[1], image.shape[0] // 4 - rnd(firstDerMax * 150)), (0, 64, 64), 1)
cv.line(image3, (0, image.shape[0] // 4 + rnd(firstDerMax * firstDerThreshold * 150)), (image.shape[1], image.shape[0] // 4 + rnd(firstDerMax * firstDerThreshold * 150)), (0, 64, 64), 1)
cv.line(image3, (0, image.shape[0] // 4 - rnd(firstDerMax * firstDerThreshold * 150)), (image.shape[1], image.shape[0] // 4 - rnd(firstDerMax * firstDerThreshold * 150)), (0, 64, 64), 1)
cv.line(image3, (0, image.shape[0] // 2 + rnd(secondDerThreshold * 3000)), (image.shape[1], image.shape[0] // 2 + rnd(secondDerThreshold * 3000)), (0, 0, 64), 1)
cv.line(image3, (0, image.shape[0] // 2 - rnd(secondDerThreshold * 3000)), (image.shape[1], image.shape[0] // 2 - rnd(secondDerThreshold * 3000)), (0, 0, 64), 1)
cv.line(image3, (0, 3 * image.shape[0] // 4 + rnd(thirdDerThreshold * 20000)), (image.shape[1], 3 * image.shape[0] // 4 + rnd(thirdDerThreshold * 20000)), (64, 0, 64), 1)
cv.line(image3, (0, 3 * image.shape[0] // 4 - rnd(thirdDerThreshold * 20000)), (image.shape[1], 3 * image.shape[0] // 4 - rnd(thirdDerThreshold * 20000)), (64, 0, 64), 1)

for i in range(len(filteredSlopes)): # Checks if first derivative is in bounds and draws remaining slopes
    if abs(firstDerivative[filteredSlopes[i][1]][0]) > firstDerThreshold * firstDerMax:
        curIndex = int(round(firstDerivative[filteredSlopes[i][1]][1] - firstDerSmplSize // 2))
        #cv.line(image2, (curIndex, rnd(smoothEdge[curIndex]) - 0), (curIndex + firstDerSmplSize, int(round(rnd(smoothEdge[curIndex]) - 0 - (firstDerivative[filteredSlopes[i][1]][0] * firstDerSmplSize)))), (0, 255, 0), 2)
        filteredSlopes2.append(firstDerivative[filteredSlopes[i][1]][0])

regResAbs = list(map(abs, filteredSlopes2)) # Gets array of remaining slopes in degrees
regResAbs = [round(math.atan(x) * 180 / np.pi, 3) for x in regResAbs]

mean = np.mean(regResAbs) # Gets mean and standard deviation
sigma = np.std(regResAbs)

plt.hist(regResAbs) # Plots histogram
plt.title("Histogram of Filtered Slopes")
plt.text(3,2,"Mean: "+str(round(mean,3)))
plt.text(2,2,"SD: "+str(round(sigma,3)))

print("Max Angle:", round(math.atan(firstDerMax) * 180 / np.pi, 2)) # Prints more results
print("Mean: "+str(round(mean,2)))
print("SD: "+str(round(sigma,2)))
print("Coefficient of variation: "+ str(round(sigma/mean,2)))

cv.imshow("Detected Edges and Best-Fit Lines", image2.astype(np.uint8)) # Shows images
cv.imshow("Nth Derivitive data", image3)
plt.show()
