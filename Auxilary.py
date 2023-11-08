from Config import Config
import numpy as np
import numpy.ma as ma
import cv2 as cv
from skimage.util import random_noise

# Auxilary Functions
def roundHalfUp(a):
    """
    Function that rounds the way that matlab would. Necessary for the program to run like the matlab version
    :param a: numpy array or float
    :return: input rounded
    """
    return (np.floor(a) + np.round(a - (np.floor(a) - 1)) - 1)


def uint8(a):
    """
    This function is necessary to turn back arrays and floats into uint8.
    arr.astype(np.uint8) could be used, but it rounds differently than the
    matlab version.
    :param a: numpy array or float
    :return: numpy array or float as an uint8
    """

    a = roundHalfUp(a)
    if np.ndim(a) == 0:
        if a < 0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a > 255] = 255
        a[a < 0] = 0
    return a


def imGaussNoise(image, mean, var):
    """
       Function used to make image have static noise

       Args:
           image (numpy array): image
           mean (float): mean
           var (numpy array): var

       Returns:
            noisy (numpy array): image with noise applied
       """
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    gauss = gauss.reshape(row, col)
    noisy = image + gauss
    return noisy


def createDepthArr(img, xIdx, yIdx, d):
    """
        Gives each pixel of the image depth, it simpy dilates the depth at each keypoint

        Args:
            img (numpy array): img of size imageSizeX by imageSizeY of the fish
            xIdx (numpy array): x coordinates of the keypoints
            yIdx (numpy array): y coordinates of the keypoints
            d (numpy array): the depth of each keypoint
        Returns:
            depthImage (numpy array): img of size imageSizeX by imageSizeY with each pixel of the fish
                                        representing its depth
    """
    imageSizeY, imageSizeX = img.shape[:2]
    depthArr = np.zeros( (imageSizeY, imageSizeX) )
    depthArrCutOut = np.zeros( (imageSizeY, imageSizeX) )

    radius = 14
    for point in range(10):
        [backboneY, backboneX] = [(np.ceil(yIdx).astype(int))[point], (np.ceil(xIdx).astype(int))[point]]
        depth = d[point]
        if (backboneY <= imageSizeY-1) and (backboneX <= imageSizeX-1) and (backboneX >= 0) and (backboneY >= 0):
            depthArr[backboneY,backboneX] = depth
    kernel = np.ones(( (radius * 2) + 1, (radius * 2) + 1 ) )
    depthArr = cv.dilate(depthArr,kernel= kernel)

    depthArrCutOut[img != 0] = depthArr[img != 0]
    return depthArrCutOut

def mergeGreysExactly(grays, depths):
    """
        Function that merges grayscale images without blurring them
    :param grays: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :param depths: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :return: 2 numpy arrays of size (imageSizeY, imageSizeX) representing the merged depths and grayscale images
        also returns the indices of the fishes in the front
    """
    indicesForTwoAxis = np.indices(grays.shape[1:])

    # indicesFor3dAxis = np.argmin(ma.masked_where(depths == 0, depths), axis=0)
    # has to be masked so that you do not consider parts where there are only zeros
    indicesFor3dAxis = np.argmin(ma.masked_where( grays == 0, depths ), axis=0 )

    indices2 = indicesFor3dAxis, indicesForTwoAxis[0], indicesForTwoAxis[1]

    mergedGrays = grays[indices2]
    mergedDepths = depths[ indices2]

    return mergedGrays, mergedDepths , indices2

def mergeGreys(grays, depths):
    """
        Function that merges grayscale images while also blurring the edges for a more realistic look
    :param grays: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :param depths: numpy array of size( n_fishes, imageSizeY, imageSizeX)
    :return: 2 numpy arrays of size (imageSizeY, imageSizeX) representing the merged depths and grayscale images
    """

    # Checking for special cases
    amountOfFishes = grays.shape[0]
    if amountOfFishes == 1:
        return grays[0], depths[0]
    if amountOfFishes == 0 :
        # return np.zeros((grays.shape[1:3])), np.zeros((grays.shape[1:3]))
        return np.zeros((Config.imageSizeY, Config.imageSizeX)), \
                np.zeros((Config.imageSizeY, Config.imageSizeX))


    threshold = Config.visibilityThreshold
    mergedGrays, mergedDepths, indices = mergeGreysExactly(grays, depths)

    # Blurring the edges

    # will be used as the brightness when there is no fish underneath the edges with
    # brightness greater than the threshold
    maxes = np.max(grays, axis=0)

    # will be used as the ordered version of brightnesses greater than the threshold
    grays[grays < threshold] = 0
    graysBiggerThanThresholdMerged, _, _ = mergeGreysExactly(grays, depths)

    # applying the values to the edges
    indicesToBlurr = np.logical_and( np.logical_and( mergedGrays < threshold, mergedGrays > 0 ),
                                     graysBiggerThanThresholdMerged > 0 )
    mergedGrays[ indicesToBlurr ] = graysBiggerThanThresholdMerged[ indicesToBlurr ]
    indicesToBlurr = np.logical_and( np.logical_and( mergedGrays < threshold, mergedGrays > 0 ),
                                     maxes > 0)
    mergedGrays[ indicesToBlurr ] = maxes[indicesToBlurr]

    # NOTE: we could technically also blurr the depths?
    return mergedGrays, mergedDepths

def mergeViews(views_list):
    finalViews = []
    amount_of_cameras = len(views_list[0])
    amount_of_fish = len(views_list)
    for camera_idx in range(amount_of_cameras):
        # Getting the views with respect to each camera
        im_list = []
        depth_im_list = []
        for fish_idx in range(amount_of_fish):
            im = views_list[fish_idx][camera_idx][0]
            depth_im = views_list[fish_idx][camera_idx][1]

            im_list.append(im)
            depth_im_list.append(depth_im)

        grays = np.array(im_list)
        depths = np.array(depth_im_list)

        finalGray, finalDepth = mergeGreys(grays, depths)
        finalView = (finalGray, finalDepth)
        finalViews.append(finalView)
    return finalViews


def add_noise_static_noise(im):
    # Adding gaussian noise
    filter_size = 2 * roundHalfUp(np.random.rand()) + 3
    sigma = np.random.rand() + 0.5
    kernel = cv.getGaussianKernel(int(filter_size), sigma)
    im = cv.filter2D(im, -1, kernel)
    maxGray = max(im.flatten())
    if maxGray != 0:
        # im = im / max(im.flatten())
        im = im / 255

    else:
        im[0, 0] = 1
    im = imGaussNoise(im, (np.random.rand() * np.random.normal(50, 10)) / 255,
                      (np.random.rand() * 50 + 20) / 255 ** 2)
    # Converting Back
    if maxGray != 0:
        # im = im * (255 / max(im.flatten()))
        im = im * 255
    else:
        im[0, 0] = 0
        im = im * 255
    im = uint8(im)

    return im


def add_patchy_noise(im, fish_list):
    imageSizeY, imageSizeX = im.shape[:2]

    averageAmountOfPatchyNoise = Config.averageAmountOfPatchyNoise

    pvar = np.random.poisson(averageAmountOfPatchyNoise)
    if (pvar > 0):

        for i in range(1, int(np.floor(pvar + 1))):
            # No really necessary, but just to ensure we do not lose too many
            # patches to fishes barely visible or fishes that do not appear in the view

            idxListOfPatchebleFishes = [idx for idx, fish in enumerate(fish_list) if
                                        fish.is_valid_fish]

            # idxListOfPatchebleFishes = [idx for idx, fish in enumerate(fishVectList + overlappingFishVectList) if fish.is_valid_fish]
            amountOfPossibleCenters = len(idxListOfPatchebleFishes)
            finalVar_mat = np.zeros((imageSizeY, imageSizeX))
            amountOfCenters = np.random.randint(0, high=(amountOfPossibleCenters + 1))
            # print('amount_of_centers: ', amountOfCenters)
            for centerIdx in range(amountOfCenters):
                # y, x
                center = np.zeros((2))
                shouldItGoOnAFish = True if np.random.rand() > .5 else False
                if shouldItGoOnAFish:
                    fish = (fish_list)[idxListOfPatchebleFishes[centerIdx]]

                    # fish = (fishVectList + overlappingFishVectList)[ idxListOfPatchebleFishes[centerIdx] ]

                    boundingBox = fish.boundingBox

                    # boundingBox = boundingBoxList[idxListOfPatchebleFishes[centerIdx]]

                    center[0] = (boundingBox.getHeight() * (np.random.rand() - .5)) + boundingBox.getCenterY()
                    center[1] = (boundingBox.getWidth() * (np.random.rand() - .5)) + boundingBox.getCenterX()
                    center = center.astype(int)
                    # clip just in case we went slightly out of bounds
                    center[0] = np.clip(center[0], 0, imageSizeY - 1)
                    center[1] = np.clip(center[1], 0, imageSizeX - 1)

                else:
                    center[0] = np.random.randint(0, high=imageSizeY)
                    center[1] = np.random.randint(0, high=imageSizeX)

                zeros_mat = np.zeros((imageSizeY, imageSizeX))
                zeros_mat[int(center[0]) - 1, int(center[1]) - 1] = 1
                randi = (2 * np.random.randint(5, high=35)) + 1
                se = cv.getStructuringElement(cv.MORPH_ELLIPSE, (randi, randi))
                zeros_mat = cv.dilate(zeros_mat, se)
                finalVar_mat += zeros_mat

            im = im / 255
            # gray_b = imnoise(gray_b, 'localvar', var_mat * 3 * (np.random.rand() * 60 + 20) / 255 ** 2)
            im = random_noise(im, mode='localvar', local_vars=(finalVar_mat * 3 * (
                    np.random.rand() * 60 + 20) / 255 ** 2) + .00000000000000001)
            im = im * 255
    return im