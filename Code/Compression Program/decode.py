import os
import sys

import cv2
import numpy as np
import math

from zigzag import *

QUANT = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],[14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],[49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]])

def getActualImage(array,h,w):
    # loop for constructing intensity matrix form frequency matrix (IDCT and all)
    i = 0
    j = 0
    k = 0

    # initialisation of compressed image

    pImage = np.zeros((h, w))

    while i < h:
        j = 0
        while j < w:
            blockArrays = array[i:i + 8, j:j + 8]
            block = inZ(blockArrays.flatten(), int(8), int(8))
            #dequantasation
            deQ = np.multiply(block, QUANT)
            #inverse dct
            pImage[i:i + 8, j:j + 8] = cv2.idct(deQ)
            j = j + 8
        i = i + 8

    # clamping to  8-bit max-min values
    pImage[pImage > 255] = 255
    pImage[pImage < 0] = 0

    # compressed image is written into compressed_image.mp file

    return pImage


def decompression(deImage, height, width):
    imageDecm = deImage.strip('][').split(', ')

    imageDecm = np.array(imageDecm)
    imageDecm = imageDecm.astype(np.float)

    return getActualImage(np.array([[imageDecm[i + j * width] for i in range(width)] for j in range(height)]), height, width)

