import os

import cv2 as cv
from matplotlib import pyplot as plt

def RLE(anyString):
    anyString_lower = anyString.upper()
    finalString = ""
    count = 1
    for x in range(0,len(anyString_lower)):
        if x+1 < len(anyString_lower):
            if anyString_lower[x] == anyString_lower[x + 1]:
                count += 1
            else:
                finalString += str(count) + anyString_lower[x]
                count = 1
        else:
            finalString += str(count) + anyString_lower[x]

    return finalString


print(RLE("00000011001100101010100101010010100000001111100010011010"))