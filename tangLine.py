import numpy
import cv2

def findTangent(home, end, slope, imageSize):
    imageX , imageY = imageSize[0], imageSize[1] # imageSize is an array of two .shape values (x,y)
    homeX , homeY = home[0], home[1] # home is an array of two values (x,y) to dictate where the starting point is
    # end is one value (x) to dictate where the ending vertical is
    endY = homeY - (slope * end)
    return int(numpy.round(endY))