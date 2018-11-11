import cv2, keyboard
import numpy as np
from sys import argv
from pprint import pprint
from tqdm import tqdm
import csv

'''
1. This visualizer module has some regions left intentionally modifiable to accommodate
any changes you deem necessary based on your application or personal preference.

2. Additionally, lines that may appear obfuscated due to their pythonic syntax have
verbose versions right above them that elicit the same functionality.

3. For quitting the display, please use the 'q' key on your keyboard. You may also
choose to terminate the program directly through other means.

4. Kindly maintain the directory structure, as this block of code is not robust to
alterations.
'''
def readCSV(filename):

    allImages = list()
    colors = [(0,0,255),(255,0,0),(0,255,0),(255,255,0),(255,0,255),(0,255,255),(128,0,128)]
    lines = None

    try:
        f =  open(filename,'r')
        new_f = open("../data/cutted_data/pic_info.csv","w")
        first_line = f.readline()
        new_f.write(first_line)
        cvs_writer = csv.writer(new_f,delimiter = ',')
        csv_reader = csv.reader(f,delimiter=',')

    except Exception as e:
        print("\n\nFile not found")
        print("Please place your file in the same directory or provide an absolute path")
        print("In the event you're using data.csv, please place it in the same directory as this program file")
        exit(0)

    loopIndex=0
    activeState = True

    for tokens in tqdm(csv_reader,ncols=700):
        # Get current image
        fileName = tokens[0]
        # Read image
        im = cv2.imread("../data/raw_data/"+fileName, cv2.IMREAD_GRAYSCALE)

        cuttedImg,start_point = cutImg(im)
        adjustBox(start_point,tokens,cvs_writer)
        new_f.flush()

        # Show keypoints
        cv2.imwrite("../data/cutted_data/"+fileName,cuttedImg)

        loopIndex += 1
    new_f.close()

def cutImg(originImg):
    #originImg = cv2.GaussianBlur(originImg,(21,21),0)
    smallImg = cv2.resize(originImg,(0,0),fx = 0.1,fy = 0.1)
    cannyImg = cv2.Canny(smallImg,180,200)

    height,width= cannyImg.shape
    maxHeight = 0
    minHeight = height
    maxWidth = 0
    minWidth = width
    for x in range(0,height):
        for y in range(0,width):
            if cannyImg[x,y] == 255:
                maxHeight = max(maxHeight,x)
                minHeight = min(minHeight,x)
                maxWidth = max(maxWidth,y)
                minWidth = min(minWidth,y)

    originHeight,originWidth = originImg.shape
    maxHeight = (maxHeight/height)*originHeight
    maxHeight = int(maxHeight)
    minHeight = (minHeight/height)*originHeight
    minHeight = int(minHeight)
    maxWidth = (maxWidth/width)*originWidth
    maxWidth = int(maxWidth)
    minWidth = (minWidth/width)*originWidth
    minWidth = int(minWidth)
    return originImg[minHeight:maxHeight,minWidth:maxWidth],(minHeight,minWidth)

def adjustBox(start_point,tokens,cvs_writer):
    sixBoxes = tokens[1:]
    new_boxes = tokens[0:1]
    for i in range(0,len(sixBoxes),10):
        oneBox = sixBoxes[i:i+10]
        for i in range(1,9,2):
            oneBox[i] = int(oneBox[i])- start_point[0]
            oneBox[i+1] = int(oneBox[i+1]) - start_point[1]
        new_boxes += oneBox
    cvs_writer.writerow(new_boxes)


def main():
    Guide = ["The program by default attempts to read a file called data.csv", "If you chose to call your file something else, please provide it's name followed by .csv as a command line argument",
            "eg:- visualizer.py myfile.csv","\n",
            "The navigation keys have been mapped as follows:",
            "'Q' = terminates the program",
            "'A' = previous image",
            "'D' = next image",
            "Note - pressing any other key will refresh the image"]
    for guidelines in Guide:
        print(guidelines)

    readCSV("../data/raw_data/combined-training.csv" if len(argv)==1 else argv[1])

if __name__ == '__main__':
    main()
