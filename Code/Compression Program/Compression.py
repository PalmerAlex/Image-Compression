import sys
import time

import cv2 as cv
from matplotlib import pyplot as plt
import os
import numpy as np
import argparse
import os
import math
import numpy as np
from utils import *
from scipy import fftpack
from PIL import Image
from huffman import HuffmanTree
import os
from PIL import Image
import cv2
import numpy as np
import scipy.fftpack as fftpack
import zlib

from zigzag import zigzag
import warnings
warnings.filterwarnings("ignore")
prev_16 = lambda x: x >> 4 << 4
bSize = 8

import matplotlib.image as mpimg
chosen_file = ""
image_dict = {
}


# plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
# plt.show()


def onRun():
    get_file_names()
    menu()


def get_file_names():
    x = 0
    directory = os.fsencode("images_before_compression")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".bmp"):
            x += 1
            image_dict[x] = filename
            continue
        else:
            continue
    x = 0
    directory = os.fsencode("images_after_compression")
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".bmp"):
            x += 1
            continue
        else:
            continue


def display_files():
    print(f'{"LoadFile":=^20}')
    for key in image_dict:
        print(f'{key:5}', image_dict[key])
    user_input = input("Select file, (the number surrounded by [ ]")
    global chosen_file
    chosen_file = image_dict[int(user_input)]
    loadFile(chosen_file)
    return True


def loadFile(string):
    global image
    global name
    image = cv.imread('images_before_compression/'+string)
    name = string
def Encoding_Quantitisation_Matrix(orig, quant):
    # import code
    # code.interact(local=vars())
    return (orig / quant).astype(np.int)


def Decoding_Quantitsation_Matrix(orig, quant):
    return (orig * quant).astype(float)


def encoding_dct_matrix(orig, bx, by):
    new_shape = (
        orig.shape[0] // bx * bx,
        orig.shape[1] // by * by,
        3
    )
    new = orig[
        :new_shape[0],
        :new_shape[1]
    ].reshape((
        new_shape[0] // bx,
        bx,
        new_shape[1] // by,
        by,
        3
    ))

    QUANT = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
    img1 = cv2.imread('images_before_compression/'+name, -1)

    # splits image into three colour channels ~~ open cv takes BGR not RGB
    print(f'{"Splitting Image into three channels":=^40}')
    time.sleep(1)
    Blue, Green, Red = cv2.split(img1)

    [height, width] = Green.shape  # gets height and width of image, blue green or red all have same heights

    # creates boxes with heights and assign widths
    heightOne = height
    widthOne = width
    height = np.float32(height)
    width = np.float32(width)
    boxHeight = math.ceil(height / bSize)
    boxHeight = np.int32(boxHeight)
    boxWidth = math.ceil(width / bSize)
    boxWidth = np.int32(boxWidth)
    Height = bSize * boxHeight
    Width = bSize * boxWidth
    # done creating boxes and assigning widths
    print(f'{"Padding Image":=^40}')
    time.sleep(1)
    # Image pad
    PaddedImageBlue = np.zeros((Height, Width))
    PaddedImageGreen = np.zeros((Height, Width))
    PaddedImageRed = np.zeros((Height, Width))
    # done padding with empty arrays

    # fill image with colours
    PaddedImageBlue[0:heightOne, 0:widthOne] = Blue[0:heightOne, 0:widthOne]
    PaddedImageGreen[0:heightOne, 0:widthOne] = Green[0:heightOne, 0:widthOne]
    PaddedImageRed[0:heightOne, 0:widthOne] = Red[0:heightOne, 0:widthOne]
    # done filling images with colours from imported image

    # display image in openCV before any compression (including padding)
    print(f'{"Applying DCT and quantisation matrix":=^40}')
    ImageWithPaddingBeforeCompression = cv2.merge([PaddedImageBlue, PaddedImageGreen, PaddedImageRed])
    for y in range(boxHeight):
        rowOne = y * bSize
        rowTwo = rowOne + bSize

        for z in range(boxWidth):
            colOne = z * bSize
            colTwo = colOne + bSize

            BluePadded = PaddedImageBlue[rowOne: rowTwo, colOne: colTwo]
            GreenPadded = PaddedImageGreen[rowOne: rowTwo, colOne: colTwo]
            RedPadded = PaddedImageRed[rowOne: rowTwo, colOne: colTwo]

            # does dct
            BlueDCT = cv2.dct(BluePadded)
            GreenDCT = cv2.dct(GreenPadded)
            RedDCT = cv2.dct(RedPadded)
            # finish dct

            # start normalisations of dct
            BlueDCTNormalised = np.divide(BlueDCT, QUANT).astype(int)
            GreenDCTNormalised = np.divide(GreenDCT, QUANT).astype(int)
            RedDCTNormalised = np.divide(RedDCT, QUANT).astype(int)
            # end normalisation of dct

            # reodering through zigzag
            BlueReordering = zigzag(BlueDCTNormalised)
            GreenReordering = zigzag(GreenDCTNormalised)
            RedReordering = zigzag(RedDCTNormalised)
            # finishing zizag reordring

            # reshaping starting
            BlueReshaping = np.reshape(BlueReordering, (bSize, bSize))
            GreenReshaping = np.reshape(GreenReordering, (bSize, bSize))
            RedReshaping = np.reshape(RedReordering, (bSize, bSize))
            # reshaping done
            # applying change to padded channels
            PaddedImageBlue[rowOne: rowTwo, colOne: colTwo] = BlueReshaping
            PaddedImageGreen[rowOne: rowTwo, colOne: colTwo] = GreenReshaping
            PaddedImageRed[rowOne: rowTwo, colOne: colTwo] = RedReshaping
            # finished applying change to padded channels
    print(f'{"DCT complete":=^40}')
    DCTtransformOfImage = cv2.merge([PaddedImageBlue, PaddedImageGreen, PaddedImageRed])
    cv2.imwrite('images_with_dct/{}encoded.bmp'.format(name), DCTtransformOfImage)
    cv2.imwrite('images_with_dct/{}encoded_as_uint8.bmp'.format(name), np.uint8(DCTtransformOfImage))
    print(f'{"Saved DCT of {} to file":=^40}'.format(name))
    time.sleep(2)
    return fftpack.dctn(new, axes=[1,3], norm='ortho')


def decode_dct(orig, bx, by):
    print(f'{"Decoding DCT":=^40}')
    time.sleep(5)
    print(f'{"DCT decoding complete":=^40}')
    return fftpack.idctn(orig, axes=[1,3], norm='ortho'
    ).reshape((
        orig.shape[0]*bx,
        orig.shape[2]*by,
        3
    ))


def encoding_compressed_image(x):
    print(f'{"Lossless Compression started":=^40}')
    time.sleep(1)
    print(f'{"Lossless Compression ended":=^40}')
    return zlib.compress(x.astype(np.int8).tobytes())


def decoding_compressed_image(orig, shape):
    print(f'{"Lossless decompression started":=^40}')
    time.sleep(5)
    print(f'{"Lossless deompression complete":=^40}')
    return np.frombuffer(zlib.decompress(orig), dtype=np.int8).astype(float).reshape(shape)


def compress(blocksize,quality):
    im = image
    quants = [quality]  # [0.5, 1, 2, 5, 10]
    blocks = []  # [(2, 8), (8, 8), (16, 16), (32, 32), (200, 200)]
    blocks.append(blocksize)

    for qscale in quants:
        for bx, by in blocks:
            quant = (
                (np.ones((bx, by)) * (qscale * qscale))
                .clip(-100, 100)  # to prevent clipping
                .reshape((1, bx, 1, by, 1))
            )



            #encoding using quality factor block size and the image
            enc = encoding_dct_matrix(im, bx, by)
            encq = Encoding_Quantitisation_Matrix(enc, quant)
            encz = encoding_compressed_image(encq)
            print(f'{"Compressed image saved to memory":=^40}')

            #decoding using old encoding lossless compression
            decz = decoding_compressed_image(encz, encq.shape)
            print(f'{"Compressed image read from Memory":=^40}')
            decq = Decoding_Quantitsation_Matrix(encq, quant)

            dec = decode_dct(decq, bx, by)
            cv2.imwrite("images_after_compression/" +name +"_recompressed_quant_{}_block_{}x{}.bmp".format(qscale, bx, by), dec.astype(np.uint8))


            # closing all open windows
            cv2.destroyAllWindows()
            MSE = round(np.square(np.subtract(dec, im)).mean(),3)
            compression_percentage = str(round(sys.getsizeof(np.uint8(im))/sys.getsizeof(encz) * 100)) + "%"
            compression_ratio = str(round(sys.getsizeof(np.uint8(im))/sys.getsizeof(encz))) + ":1"
            originalFileSize = str(sys.getsizeof(np.uint8(im)) /1000000) + "MB"
            compressedFileSize = str(sys.getsizeof(encz) /1000000) + "MB"
            decompressedFileSize = str(sys.getsizeof(np.uint8(dec)) /1000000) + "MB"
            print(f'{"file size of original image":=^40}')
            print(f"|{originalFileSize:^40}|")
            print(f'{"file size of decompressed image":=^40}')
            print(f"|{decompressedFileSize:^40}|")
            print(f'{"file size of compressed image":=^40}')
            print(f"|{compressedFileSize:^40}|")
            print(f'{"compression percentage":=^40}')
            print(f"|{compression_percentage:^40}|")
            print(f'{"compression ratio":=^40}')
            print(f"|{compression_ratio:^40}|")
            print(f'{"MSE":=^40}')
            print(f"|{MSE:^40}|")

            plt.title("Decompressed Image")
            plt.imshow(cv.cvtColor(dec.astype(np.uint8), cv.COLOR_BGR2RGB))
            plt.show()
            cv2.waitKey(0)



def CompressMenu():
    blocksize = []
    qualityfactor = 0
    print(f'{"Compression-Menu":=^20}')

    print(f'{"0":5} : 4x4')
    print(f'{"1":5} : 8x8')
    print(f'{"2":5} : 16x16')
    print(f'{"3":5} : 32x32')
    user_input = int(input("Select Block Size: "))
    if user_input == 0:
        blocksize = (4,4)
    if user_input == 1:
        blocksize = (8,8)
    if user_input == 2:
        blocksize = (16,16)
    if user_input == 3:
        blocksize = (32,32)
    print(f'{"Block size selected:":=^20}',format(blocksize))
    print(f'{"Compression-Menu":=^20}')
    qualityfactor = int(input("Enter Quality Factor (0 - 10) (0 means no compression, no data loss) (10 means almost max compression lots of data loss) "))
    print(f'{"Quality factor selected:":=^20}', format(qualityfactor))
    print(f'{"Compression Started":=^40}')
    compress(blocksize,qualityfactor)

def menu():
    is_loaded = False
    while True:
        print(f'{"Menu":=^20}')
        if is_loaded:
            print(f'{"0":5} : Compress')
            #print(f'{"1":5} : Save')
            print(f'{"2":5} : Display Image')
            print(f'{"3":5} : Display File Size')
            print(f'{"4":5} : Load Another Image')
        else:
            print(f'{"5":5} : Load')
        print(f'{"6":5} : Quit')
        user_input = int(input("Select process, (the number surrounded by [ ]"))

        if user_input == 0 and is_loaded != False:
           CompressMenu()
        if user_input == 1 and is_loaded != False:
            save_to_file()
        if user_input == 2 and is_loaded != False:
            display_image()
        if user_input == 3 and is_loaded != False:
            print(get_file_size())
        if user_input == 4 and is_loaded != False:
            is_loaded = display_files()
        if user_input == 5 and is_loaded == False:
            is_loaded = display_files()
        if user_input == 6:
            quit()

def save_to_file():
    print("save")
def display_image():
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show()
def get_file_size():
    global filesize
    filesize = round(os.path.getsize('images_before_compression/'+name) / 1048576,2)
    return "File size is :  "+str(filesize)+ " MB"
onRun()







