from Config import Config
import numpy as np
import numpy.ma as ma
import cv2 as cv


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


    threshold = 25
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
