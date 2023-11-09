import numpy as np
import imageio
import cv2 as cv
from Programs.Config import Config

# INPUTS
imageHeight, imageWidth = Config.imageSizeY, Config.imageSizeX

# Constants
red = [0, 0, 255]
green = [0, 255, 0]
blue = [255, 0, 0]

def turn_text_file_to_data(file):
    f = open(file,'r')

    keypointsListContainer = [[]]
    boxListContainer = [[]]
    # iterating through the lines in the text file
    for line in f:
        keypointsArr = np.zeros((12, 3))
        words = line.split()
        keypoints = words[5:]

        viewIdx = int(words[0])
        keypointsList = keypointsListContainer[viewIdx]
        boxList = boxListContainer[viewIdx]

        for x in range(12):
            step = 3
            xInFormat = x * step
            keypointsArr[x, :] = [float(keypoints[xInFormat]) * imageWidth,
                                  float(keypoints[xInFormat + 1]) * imageHeight,
                                  float(keypoints[xInFormat + 2])]
        keypointsList.append(keypointsArr)

        bBox = words[1:5]
        [x, y, w, h] = [int(np.ceil(float(bBox[0]) * imageWidth)), int(np.ceil(float(bBox[1]) * imageHeight)),
                        int(np.ceil(float(bBox[2]) * imageWidth)), int(np.ceil(float(bBox[3]) * imageHeight))]
        [tx, bx, ty, by] = [x + int(np.ceil(w / 2)), x - int(np.ceil(w / 2)), y + int(np.ceil(h / 2)),
                            y - int(np.ceil(h / 2))]
        bBoxConverted = [tx, bx, ty, by]
        boxList.append(bBoxConverted)

    return keypointsListContainer, boxListContainer

def draw_boxes(img, boxList):
    amountOfBoxes = len(boxList)
    for boxIdx in range(amountOfBoxes):
        box = boxList[boxIdx]
        [tx, bx, ty, by] = box
        # Converting back to the image size
        # tx, bx = tx * imageWidth, bx * imageWidth
        # ty, by = ty * imageHeight, by * imageHeight
        [tx, bx] = np.clip(np.array([tx, bx]), 0, imageWidth - 1)
        [ty, by] = np.clip(np.array([ty, by]), 0, imageHeight - 1)

        img[ty, bx:tx, :] = red
        img[by, bx:tx, :] = red
        img[by:ty, bx, :] = red
        img[by:ty, tx, :] = red

    return img

def draw_keypoints(img, keypointsList):
    amountOfKeypoints = len(keypointsList)
    for keypointsIdx in range(amountOfKeypoints):
        keypoints = keypointsList[keypointsIdx]
        for pointIdx, point in enumerate(keypoints):
            [col, row] = np.floor(point[:2]).astype(int)
            vis = point[2]
            if pointIdx < 10:
                # It is a backbone point
                if vis :
                    img[row, col] = green
                else:
                    img[row, col] = blue
            else:
                # It is an eye
                if vis:
                    img[row, col] = red
                else:
                    img[row, col] = blue
    return img

def draw_annotations_on_images(imageList, keypointsListContainer, boxListContainer):
    amountOfViews = len(imageList)
    for viewIdx in range(amountOfViews):
        img = imageList[viewIdx]


        keypointsList = keypointsListContainer[viewIdx]
        boxList = boxListContainer[viewIdx]

        img = draw_keypoints(img, keypointsList)
        img = draw_boxes(img, boxList)

        imageList[viewIdx] = img

    return imageList


def draw_YOLO_pose_annotations(imageFilePath):
    labelsFolder = 'labels/'
    dataFolder = 'data/'
    rest = imageFilePath[12:]
    labelsFilePath = dataFolder + labelsFolder + rest[:-3] + 'txt'
    keypointsListContainer, boundingBoxListContainer = turn_text_file_to_data(labelsFilePath)

    image = imageio.imread(imageFilePath)
    rgb = np.zeros((image.shape[0], image.shape[1], 3))
    for channel in range(3): rgb[..., channel] = image
    imageList = [rgb]

    imageListWithAnnotations = draw_annotations_on_images(imageList, keypointsListContainer, boundingBoxListContainer)

    catResults = np.concatenate(imageListWithAnnotations, axis=0)

    return catResults

# Warning: This program currently only works with the parent directory set to data
imageFilePath = 'data/images/train/zebrafish_000000.png'
result = draw_YOLO_pose_annotations(imageFilePath)
cv.imwrite('test.png', result)










