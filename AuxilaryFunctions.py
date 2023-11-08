from Config import Config
import random
import numpy as np
import cv2 as cv
import imageio
from scipy.io import loadmat
from construct_model import f_x_to_model, f_x_to_model_centered, f_x_to_model_bigger
from skimage.util import random_noise
from FunctionsFile import *

# INPUTS
# imageSizeX, imageSizeY = 640, 640
imageSizeY, imageSizeX = Config.imageSizeY, Config.imageSizeX
theta_array = loadmat('generated_pose_all_2D_50k.mat')
theta_array = theta_array['generated_pose']

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
        if a <0:
            a = 0
        if a > 255:
            a = 255
    else:
        a[a>255]=255
        a[a<0]=0
    return a

def imGaussNoise(image,mean,var):
    """
       Function used to make image have static noise

       Args:
           image (numpy array): image
           mean (float): mean
           var (numpy array): var

       Returns:
            noisy (numpy array): image with noise applied
       """
    row,col= image.shape
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    gauss = gauss.reshape(row,col)
    noisy = image + gauss
    return noisy


# Rotate along x axis. Angles are accepted in radians
def rotx(angle):
    M = np.array([[1, 0, 0],  [0, np.cos(angle), -np.sin(angle)],  [0, np.sin(angle), np.cos(angle)]])
    return M

# Rotate along y axis. Angles are accepted in radians
def roty(angle):
    M = np.array([[np.cos(angle), 0, np.sin(angle)],  [0, 1, 0],  [-np.sin(angle), 0, np.cos(angle)]])
    return M

# Rotate along z axis. Angles are accepted in radians
def rotz(angle):
    M = np.array([[np.cos(angle), -np.sin(angle), 0],  [np.sin(angle), np.cos(angle), 0],  [0, 0, 1]])
    return M

def x_seglen_to_3d_points(x, seglen):
    """
        Function that turns the x vector into the 3d points of the fish
    :param x:
    :param seglen:
    :return:
    """
    hp = x[0: 2]
    dt = x[2: 11]
    pt = np.zeros((2, 10))
    theta = np.zeros((9, 1))
    theta[0] = dt[0]
    pt[:, 0] = hp

    for n in range(0, 9):
        R = np.array([[np.cos(dt[n]), -np.sin(dt[n])], [np.sin(dt[n]), np.cos(dt[n])]])
        if n == 0:
            vec = np.matmul(R, np.array([seglen, 0], dtype=R.dtype))
        else:
            vec = np.matmul(R, vec)
            theta[n] = theta[n - 1] + dt[n]
        pt[:, n + 1] = pt[:, n] + vec

    # Now calculating the eyes
    size_lut = 49
    size_half = (size_lut + 1) / 2

    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])

    d_eye = seglen

    XX = size_lut
    YY = size_lut
    ZZ = size_lut

    c_eyes = 1.9
    c_belly = 0.98
    c_head = 1.04
    canvas = np.zeros((XX, YY, ZZ))

    theta, gamma, phi = x[2], 0, 0

    R = rotz(theta) @ roty(phi) @ rotx(gamma)

    # Initialize points of the ball and stick model in the canvas
    pt_original = np.zeros((3, 3))
    # pt_original_1 is the mid-point in Python's indexing format
    pt_original[:, 1] = np.array([np.floor(XX / 2) + dh1, np.floor(YY / 2) + dh2, np.floor(ZZ / 2)])
    pt_original[:, 0] = pt_original[:, 1] - np.array([seglen, 0, 0], dtype=pt_original.dtype)
    pt_original[:, 2] = pt_original[:, 1] + np.array([seglen, 0, 0], dtype=pt_original.dtype)

    eye1_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye1_c = eye1_c - pt_original[:, 1, None]
    eye1_c = np.matmul(R, eye1_c) + pt_original[:, 1, None]

    eye2_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]],
                       [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] - d_eye / 2],
                       [pt_original[2, 1] - seglen / 8]], dtype=pt_original.dtype)
    eye2_c = eye2_c - pt_original[:, 1, None]
    eye2_c = np.matmul(R, eye2_c) + pt_original[:, 1, None]

    eye1_c[0] = eye1_c[0] - (size_half - 1) + pt[0, 1]
    eye1_c[1] = eye1_c[1] - (size_half - 1) + pt[1, 1]
    eye2_c[0] = eye2_c[0] - (size_half - 1) + pt[0, 1]
    eye2_c[1] = eye2_c[1] - (size_half - 1) + pt[1, 1]

    pt = np.concatenate([pt, eye1_c[0: 2], eye2_c[0: 2]], axis=1)

    return pt

def addBoxes(pt, padding= .7):
    """
        Function that adds boxes to the fish given its backbone points
    :param pt:
    :param padding:
    :return:
    """
    number_of_eyes = 2

    threshold = Config.minimumSizeOfBox
    # minus one assuming the boxes are in the center
    amount_of_boxes = pt.shape[1] - number_of_eyes - 1

    dimensions_of_center = 2
    dimensions_of_box = 2
    pointsAndBoxes = np.zeros(( dimensions_of_box + dimensions_of_center, amount_of_boxes))
    # TODO:vectorize
    for pointIdx in range(amount_of_boxes):
        fPX, fPY = pt[:, pointIdx]
        sPX, sPY = pt[:, pointIdx + 1]

        c = np.array( [(fPX + sPX)/ 2, (fPY + sPY)/2] )
        l = np.array( [np.abs(fPX - sPX), np.abs(fPY - sPY)] )

        # Even if is close to zero there should still be some length/width to the box
        l[l < threshold] = threshold

        # Adding the padding
        l = (padding + 1) * l

        cX, cY = c
        lX, lY = l

        pointsAndBoxes[:, pointIdx] = [cX, cY, lX, lY]

    # Now the eyes
    for eyeIdx in range(number_of_eyes):
        lenghtOfEyes = 9 # ?, maybe make it depend on seglen
        cX, cY = pt[:, -number_of_eyes + eyeIdx]
        lX, lY = lenghtOfEyes, lenghtOfEyes

        box_for_eyes = np.array([ [ cX], [cY], [lX], [lY]])
        pointsAndBoxes = np.concatenate( (pointsAndBoxes, box_for_eyes), axis = 1)
    return pointsAndBoxes

def doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
    """
        Function to check if the fish in question will crash with any of the other fishes
    :param fishVect:
    :param fishVectList:
    :return:
    """
    # Preprocessing
    fishVect = np.array(fishVect)
    if len(fishVectList) == 0: return False
    fishVectList = np.array(fishVectList)

    # concatenating to avoid repetitions
    fishVectList = np.concatenate((fishVect[np.newaxis, :], fishVectList), axis=0)

    boxesList = []
    for fishVectIdx in range(fishVectList.shape[0]):
        fishVect = fishVectList[fishVectIdx, ...]

        seglen, x = fishVect[0], fishVect[1:]
        pt = x_seglen_to_3d_points(x, seglen)
        boxes_covering_fish = addBoxes(pt)

        # Converting them to top bottom format
        boxes_covering_fish_top_bottom_format = np.copy(boxes_covering_fish)
        boxes_covering_fish_top_bottom_format[0, :] = \
            boxes_covering_fish[0, :] + (boxes_covering_fish_top_bottom_format[2, :] / 2)
        boxes_covering_fish_top_bottom_format[1, :] = \
            boxes_covering_fish[0, :] - (boxes_covering_fish_top_bottom_format[2, :] / 2)
        # For the y - dimension
        boxes_covering_fish_top_bottom_format[2, :] = \
            boxes_covering_fish[1, :] + (boxes_covering_fish_top_bottom_format[3, :] / 2)
        boxes_covering_fish_top_bottom_format[3, :] = \
            boxes_covering_fish[1, :] - (boxes_covering_fish_top_bottom_format[3, :] / 2)
        # boxes_covering_fish is in format bigX, smallX, bigY, smallY

        boxesList.append(boxes_covering_fish_top_bottom_format)
    boxesInQuestion = boxesList[0]
    boxesList = boxesList[1:]
    cat = np.concatenate(boxesList, axis=1)
    number_of_boxes_per_fish = 11

    # boxes Arrays have the first row representing the biggest X value, the second row representing the smallest X value
    # the third row the biggest Y value, the fourth row the smallest Y value

    # Checking if they crashed in the x dimension
    top_x_coors_boxes_in_question = boxesInQuestion[0,:]
    bottom_x_coors_boxes = cat[1,:]
    grid = np.array( np.meshgrid( top_x_coors_boxes_in_question, bottom_x_coors_boxes ) )
    tops, bottoms = grid[0, ...], grid[1, ...]
    points_where_top_edge_is_above_bottom_edge = tops > bottoms

    bottom_x_coors_boxes_in_question = boxesInQuestion[1,:]
    top_x_coors_boxes = cat[0,:]
    grid = np.array(np.meshgrid( bottom_x_coors_boxes_in_question , top_x_coors_boxes))
    bottoms, tops = grid[0], grid[1]
    points_where_bottom_edge_is_above_top_edge = bottoms < tops

    points_where_they_crash_in_the_x_dimension = points_where_top_edge_is_above_bottom_edge * \
                                                 points_where_bottom_edge_is_above_top_edge

    # Checking if they crashed in the y dimension
    top_x_coors_boxes_in_question = boxesInQuestion[2, :]
    bottom_x_coors_boxes = cat[3, :]
    grid = np.array(np.meshgrid(top_x_coors_boxes_in_question, bottom_x_coors_boxes))
    tops, bottoms = grid[0, ...], grid[1, ...]
    points_where_top_edge_is_above_bottom_edge = tops > bottoms

    bottom_x_coors_boxes_in_question = boxesInQuestion[3, :]
    top_x_coors_boxes = cat[2, :]
    grid = np.array(np.meshgrid(bottom_x_coors_boxes_in_question, top_x_coors_boxes))
    bottoms, tops = grid[0], grid[1]
    points_where_bottom_edge_is_above_top_edge = bottoms < tops

    points_where_they_crash_in_the_y_dimension = points_where_top_edge_is_above_bottom_edge * \
                                                 points_where_bottom_edge_is_above_top_edge

    points_where_they_crashed = points_where_they_crash_in_the_x_dimension * points_where_they_crash_in_the_y_dimension

    if np.any(points_where_they_crashed):
        #print('Invalid Fish')
        return True
    else:
        #print('Valid Fish')
        return False

def isThisAGoodFishVectList(fishVectList):
    """
        Funtion that checks whether the fishVectList has fishes that are crashing into each other
    :param fishVectList:
    :return:
    """
    boxesList = []
    for fishVectIdx in range( fishVectList.shape[0] ):
        fishVect = fishVectList[fishVectIdx, ...]

        seglen, x = fishVect[0], fishVect[1:]
        pt = x_seglen_to_3d_points(x, seglen)
        boxes_covering_fish = addBoxes(pt)

        # Converting them to top bottom format
        boxes_covering_fish_top_bottom_format = np.copy(boxes_covering_fish)
        boxes_covering_fish_top_bottom_format[0,:] = \
            boxes_covering_fish[0,:] + (boxes_covering_fish_top_bottom_format[2,:] / 2)
        boxes_covering_fish_top_bottom_format[1,:] = \
            boxes_covering_fish[0,:] - (boxes_covering_fish_top_bottom_format[2,:] / 2)
        # For the y - dimension
        boxes_covering_fish_top_bottom_format[2,:] = \
            boxes_covering_fish[1,:] + (boxes_covering_fish_top_bottom_format[3,:] / 2)
        boxes_covering_fish_top_bottom_format[3,:] = \
            boxes_covering_fish[1,:] - (boxes_covering_fish_top_bottom_format[3,:] / 2)
        # boxes_covering_fish is in format bigX, smallX, bigY, smallY

        boxesList.append(boxes_covering_fish_top_bottom_format)
    cat = np.concatenate( boxesList, axis=1 )
    number_of_boxes_per_fish = 11

    # Considering if they crash in the x dimension
    top_x_coors = cat[0,:]
    bottom_x_coors = cat[1,:]
    grid = np.array( np.meshgrid(top_x_coors, bottom_x_coors) )
    top_x_coors = grid[0, ...]
    bottom_x_coors = grid[1, ...]
    pointsWhereTheTopEdgeIsInFrontOfABottomEdges = top_x_coors > bottom_x_coors

    # now the other way around
    bottom_x_coors = cat[1,:]
    top_x_coors = cat[0,:]
    grid = np.array( np.meshgrid(bottom_x_coors, top_x_coors) )
    bottom_x_coors = grid[0, ...]
    top_x_coors = grid[1, ...]
    pointsWhereTheBottomEdgesIsBeforeATopEdge = bottom_x_coors < top_x_coors

    didTheyCrashInXArray = pointsWhereTheTopEdgeIsInFrontOfABottomEdges *  pointsWhereTheBottomEdgesIsBeforeATopEdge

    # Considering if they crashed in the y dimension
    top_y_coors = cat[2,:]
    bottom_y_coors = cat[3,:]
    grid = np.array( np.meshgrid( top_y_coors, bottom_y_coors ) )
    top_y_coors = grid[0, ...]
    bottom_y_coors = grid[1, ...]
    pointsWhereTheTopEdgeIsInFrontOfABottomEdges = top_y_coors > bottom_y_coors

    bottom_y_coors = cat[3,:]
    top_y_coors = cat[2,:]
    grid = np.array( np.meshgrid( bottom_y_coors, top_y_coors ) )
    bottom_y_coors = grid[0, ...]
    top_y_coors = grid[1, ...]
    pointsWhereTheBottomEdgesIsBeforeATopEdge = bottom_y_coors < top_y_coors

    didTheyCrashInYArray = pointsWhereTheTopEdgeIsInFrontOfABottomEdges * pointsWhereTheBottomEdgesIsBeforeATopEdge
    didTheyCrash = didTheyCrashInXArray * didTheyCrashInYArray

    # TODO: vetorize this part
    for fishIdx in range(fishVectList.shape[0]):
        startIdx = fishIdx * number_of_boxes_per_fish
        endIdx = startIdx + number_of_boxes_per_fish
        didTheyCrash[startIdx:endIdx, startIdx:endIdx  ] = 0

    didTheyCrash = np.any(didTheyCrash)
    print("Crashed :'(") if didTheyCrash else print('Safe')
    if didTheyCrash:
        return False
    else:
        return True

def are_pts_in_bounds(pts):
    """
        Checks if the pts are in within [0,imageSizeX) and [0, imageSizeY)
        pts must have the first row representing the x coors
        and the second row representing the y coors
    :param pts: numpy array
    :return: boolean
    """
    # Turning it into an int to make the definition of it being in bounds concise
    are_xs_in_bounds = np.all( np.logical_and( np.ceil( pts[0,:]) < imageSizeX, pts[0,:] >= 0 ) )
    are_ys_in_bounds = np.all( np.logical_and( np.ceil( pts[1,:]) < imageSizeY, pts[1,:] >= 0 ) )

    is_pts_in_bounds = are_xs_in_bounds and are_ys_in_bounds
    return is_pts_in_bounds

def is_atleast_one_point_in_bounds(pts):
    """
        Function makes sure atleast one of the points are within 0,imageSizeX) and [0, imageSizeY)
    :param pts: numpy array
    :return: boolean
    """
    is_one_x_in = np.logical_and(np.ceil(pts[0, :]) < imageSizeX, pts[0, :] >= 0)
    is_one_y_in = np.logical_and(np.ceil(pts[1, :]) < imageSizeY, pts[1, :] >= 0)
    is_atleast_one_in = np.any(np.logical_and(is_one_x_in, is_one_y_in))
    return is_atleast_one_in

def is_fish_on_edge(pts):
    """
        Function which detects wheter a fish in on an edge.  It is defined being on an edge
        if one of the points is in the image and if one of the points is not in the image
    :param pts: numpy array
    :return: boolean
    """
    # is_one_x_in = np.logical_and( np.ceil( pts[0,:]) < imageSizeX, pts[0,:] >= 0 )
    # is_one_y_in = np.logical_and( np.ceil( pts[1,:]) < imageSizeY, pts[1,:] >= 0 )
    # is_atleast_one_in =np.any( np.logical_and( is_one_x_in, is_one_y_in) )
    is_atleast_one_in = is_atleast_one_point_in_bounds(pts)

    # Assuming of course that atleast one is in
    is_atleast_one_outside = not are_pts_in_bounds(pts)

    is_on_edge = is_atleast_one_in and is_atleast_one_outside

    return is_on_edge

def generateRandomConfiguration(fishInView, fishInEdges, OverlappingFish):
    
    averageSizeOfFish = Config.averageSizeOfFish
    fishVectList = []

    # generating fishes that are completely in the view, based on their keypoints
    for _ in range(fishInView):
        # Loop to keep on searching in case we randomly generate a fish that was not in the view
        while True:
            # Generating random fishVect
            xVect = np.zeros((11))
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            # seglen = 7.1

            x , y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
            theta_array_idx = np.random.randint(0,500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x,y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishVect = np.zeros((12))
            fishVect[0] = seglen
            fishVect[1:] = xVect


            # Checking if it is in bounds
            pts = x_seglen_to_3d_points(xVect, seglen)
            if are_pts_in_bounds(pts) == False:
                continue

            # Checking if it interferes
            if doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
                continue

            # The fish passed all the requirements, so we can add it to the fishVectList
            fishVectList.append( fishVect )
            break

    # Generating fishes in the edges
    # TODO: might want to try a different way, perhaps with shifting the fish
    for _ in range(fishInEdges):

        while True:
            # The possible edge choices for one view are: the left edge, the right edge, the top and the bottom
            # let the numbers 0 - 3 represent these choices respectively
            edgeIdx = np.random.randint(0,4)
            xVect = np.zeros((11))

            if edgeIdx < 2:
                # We got the left or right, which means the x coordinate is constrained
                if edgeIdx == 0:
                    # We want to generate points around left edge
                    x = (np.random.rand() - .5) * averageSizeOfFish
                    # Got to add some offset if we want to generate all possible conformations
                    # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
                    y = np.random.randint(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)

                else:
                    # We want to generate a fish along the right edge
                    x = (np.random.rand() - .5) * averageSizeOfFish + imageSizeX
                    # Got to add some offset if we want to generate all possible conformations
                    # y = np.random.random_integers(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
                    y = np.random.randint(0 - averageSizeOfFish/2, imageSizeY + averageSizeOfFish/2)
            else:
                # We got the top or bottom edge, which means the y coordinate is constrained
                if edgeIdx == 2:
                    # We want to generate a fish around the top edge
                    # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    y = (np.random.rand() - .5) * averageSizeOfFish
                else:
                    # We want to generate a fish around the bottom edge
                    # x = np.random.random_integers(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    x = np.random.randint(0 - averageSizeOfFish / 2, imageSizeX + averageSizeOfFish / 2)
                    y = (np.random.rand() - .5) * averageSizeOfFish + imageSizeY
            theta_array_idx = np.random.randint(0, 500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x, y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            fishVect = np.zeros((12))
            fishVect[0] = seglen
            fishVect[1:] = xVect


            pts = x_seglen_to_3d_points(xVect, seglen)
            is_on_edge = is_fish_on_edge(pts)
            if is_on_edge:
                if not doesThisFishInterfereWithTheAquarium(fishVect, fishVectList):
                    fishVectList.append(fishVect)
                    break

    # Generating Overlapping Fishes
    # NOTE: Currently overlapping fish is defined as creating fishes that will overlap the previous fishes
    # might be better if we redefine it something like pairs of overlapping fish, or maybe we can add more pairs

    # clipping the value to stay consistent with the definition above
    OverlappingFish = min(fishInView + fishInEdges, OverlappingFish)

    # Inputs deciding what is the maximum offset when causing the fishes to overlap

    maxOverlappingOffset = Config.maxOverlappingOffset

    overLappingFishVectList = []
    # We also want to overlap the fish in the edges
    fishesToOverlapIdices = random.sample([*range(len(fishVectList))], OverlappingFish )
    for overLappingFishIdx in range(OverlappingFish):

        fishVectToOverlap = fishVectList[ fishesToOverlapIdices[overLappingFishIdx] ]
        ogSeglen = fishVectToOverlap[0]
        ogXVect = fishVectToOverlap[1:]
        ogPts = x_seglen_to_3d_points(ogXVect, ogSeglen)
        # The keypoint idx of the original fish we want to overlap
        ogFishKeypointToOverlap = np.random.randint(0, 12)
        ogPoint = ogPts[:, ogFishKeypointToOverlap]

        # The keypoint of the generated fish we want to use to cause the overlap
        genFishKeypointToOverlap = np.random.randint(0, 12)

        while True:
            # Generating the fish
            xVect = np.zeros((11))
            fishlen = (np.random.rand(1) - 0.5) * 30 + averageSizeOfFish
            idxlen = np.floor((fishlen - 62) / 1.05) + 1
            seglen = 5.6 + idxlen * 0.1
            seglen = seglen[0]
            # seglen = 7.1

            x, y = np.random.randint(0, imageSizeX), np.random.randint(0, imageSizeY)
            theta_array_idx = np.random.randint(0, 500000)
            dtheta = theta_array[theta_array_idx, :]
            xVect[:2] = [x, y]
            xVect[2] = np.random.rand(1)[0] * 2 * np.pi
            xVect[3:] = dtheta
            fishVect = np.zeros((12))
            fishVect[0] = seglen
            fishVect[1:] = xVect
            pts = x_seglen_to_3d_points(xVect, seglen)
            point = pts[:, genFishKeypointToOverlap]
            distance = ogPoint - point
            # adding some randomness
            xOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
            yOffSet = ((2 * np.random.rand()) - 1) * maxOverlappingOffset
            distance += np.array([xOffSet, yOffSet])
            # Shifting the fish to cause the overlap
            xVect[0] += distance[0]
            xVect[1] += distance[1]
            fishVect[1:] = xVect

            if not doesThisFishInterfereWithTheAquarium(fishVect, overLappingFishVectList):

                # Adding the part that at least one of the points of the fish should be visible
                # this is to stop the fish from disappearing when overlapping the fish on the edges
                # NOTE: it might be a good idea to not have the fish overlap the ones on the edges
                if is_atleast_one_point_in_bounds(pts):
                    overLappingFishVectList.append(fishVect)
                    break

    return fishVectList, overLappingFishVectList


class Fish:
    class BoundingBox:
        BoundingBoxThreshold = Config.boundingBoxThreshold

        def __init__(self, smallY, bigY, smallX, bigX):
            self.smallY = smallY
            self.bigY = bigY
            self.smallX = smallX
            self.bigX = bigX

        def getHeight(self):
            return ( self.bigY - self.smallY)

        def getWidth(self):
            return ( self.bigX - self.smallX)

        def getCenterX(self):
            return (( self.bigX + self.smallX) / 2)

        def getCenterY(self):
            return (( self.bigY + self.smallY) / 2)

        def isValidBox(self):
            height = self.getHeight()
            width = self.getWidth()

            if (height <= Fish.BoundingBox.BoundingBoxThreshold) or (width <= Fish.BoundingBox.BoundingBoxThreshold):
                return False
            else:
                return True

    def __init__(self, fishVect):
        self.seglen = fishVect[0]
        self.x = fishVect[1:]

    def draw(self):
        graymodel, pts = f_x_to_model_bigger(self.x, self.seglen, Config.randomizeFish, imageSizeX, imageSizeY)

        self.pts = pts
        self.graymodel = graymodel

        self.vis = np.zeros((pts.shape[1]))
        self.vis[ self.valid_points_masks ] = 1

        # Creating the bounding box
        nonzero_coors = np.array( np.where(graymodel > 0) )
        try:
            smallY = np.min(nonzero_coors[0,:])
            bigY = np.max( nonzero_coors[0,:])
            smallX = np.min( nonzero_coors[1,:])
            bigX = np.max( nonzero_coors[1,:])
        except:
            smallY = 0
            bigY = 0
            smallX = 0
            bigX = 0
        self.boundingBox = Fish.BoundingBox(smallY, bigY, smallX, bigX)


    @property
    def xs(self):
        return self.pts[0,:]
    @property
    def ys(self):
        return self.pts[1,:]
    @property
    def intXs(self):
        return np.ceil( self.pts[0,:] ).astype(int)
    @property
    def intYs(self):
        return np.ceil( self.pts[1,:]).astype(int)
    @property
    def valid_points_masks(self):
        xs = self.intXs
        ys = self.intYs
        xs_in_bounds = (xs < imageSizeX) * (xs >= 0)
        ys_in_bounds = (ys < imageSizeY) * (ys >= 0)
        return xs_in_bounds * ys_in_bounds

    def amount_of_vis_points(self):
        val_xs = self.pts[0,:][self.valid_points_masks]
        return val_xs.shape[0]

    @property
    def is_valid_fish(self):
        if (self.amount_of_vis_points() >= 1) and self.boundingBox.isValidBox():
            return True
        else:
            return False


def drawAquarium(fishVectList, overlappingFishVectList = []):
    pts_list = []
    overlap_pts_list = []
    
    fish_list = []
    overlap_fish_list = []
    grays = []
    for fishVect in fishVectList:
        fish = Fish(fishVect)
        fish.draw()
        fish_list.append(fish)
        pts_list.append(fish.pts)
        grays.append(fish.graymodel)

    # Merging the grays, just for testing right now
    canvas = np.zeros((imageSizeY, imageSizeX))
    for gray in grays:
        canvas += gray

    overlappingGrays = []
    # now drawing the fishes in the other plane
    for fishVect in overlappingFishVectList:

        fish = Fish(fishVect)
        fish.draw()
        pts = fish.pts
        graymodel = fish.graymodel

        # updating the vis
        xs = fish.intXs
        ys = fish.intYs
        mask = fish.valid_points_masks
        values_in_canvas = canvas[ ys[mask], xs[mask]]
        mask_mask_blocked_pts = values_in_canvas > 0
        # The next 4 lines are necessary because python does not allow references, 5th is for clarity
        # TODO: condense this
        tempVis = fish.vis
        tempVis2 = tempVis[mask]
        tempVis2[mask_mask_blocked_pts] = 0
        tempVis[mask] = tempVis2
        fish.vis = tempVis
        #fish.vis[mask][mask_mask_blocked_pts] = 0

        overlap_fish_list.append( fish )

        overlap_pts_list.append(pts)
        overlappingGrays.append(graymodel)

    # Merging the grays, just for testing right now
    overlappingCanvas = np.zeros((imageSizeY, imageSizeX))
    for gray in overlappingGrays:
        overlappingCanvas += gray

    # Blurring, the threshold determines at what brightness is a fish considered solid
    threshold = Config.visibilityThreshold
    canvas[ (canvas < threshold) * (overlappingCanvas > 0) ] = \
        overlappingCanvas[ (canvas < threshold) * (overlappingCanvas > 0) ]
    # Merging the two planes now
    canvas[ canvas == 0] = overlappingCanvas[ canvas == 0]

    return canvas, fish_list, overlap_fish_list

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

def add_patchy_noise(im, fish_list, overlap_fish_list):
    imageSizeY, imageSizeX = im.shape[:2]

    averageAmountOfPatchyNoise = Config.averageAmountOfPatchyNoise

    pvar = np.random.poisson(averageAmountOfPatchyNoise)
    if (pvar > 0):

        for i in range(1, int(np.floor(pvar + 1))):
            # No really necessary, but just to ensure we do not lose too many
            # patches to fishes barely visible or fishes that do not appear in the view

            idxListOfPatchebleFishes = [idx for idx, fish in enumerate(fish_list + overlap_fish_list) if
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
                    fish = (fish_list + overlap_fish_list)[idxListOfPatchebleFishes[centerIdx]]

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

def save_annotations(frame_idx, fish_list):
    biggestIdx4TrainingData = Config.biggestIdx4TrainingData
    dataDirectory = Config.dataDirectory

    subFolder = 'train/' if frame_idx < biggestIdx4TrainingData else 'val/'
    labelsPath = dataDirectory + '/' + 'labels/' + subFolder
    strIdxInFormat = format(frame_idx, '06d')
    filename = 'zebrafish_' + strIdxInFormat + '.txt'
    labelsPath += filename

    # Creating the annotations
    f = open(labelsPath, 'w')

    for fish in (fish_list):
        # for fish in (fishVectList + overlappingFishVectList):
        boundingBox = fish.boundingBox

        # Should add a method to the bounding box, boundingBox.isSmallFishOnEdge()
        if fish.is_valid_fish:
            f.write(str(0) + ' ')
            f.write(str(boundingBox.getCenterX() / imageSizeX) + ' ' + str(boundingBox.getCenterY() / imageSizeY) + ' ')
            f.write(str(boundingBox.getWidth() / imageSizeX) + ' ' + str(boundingBox.getHeight() / imageSizeY) + ' ')

            xArr = fish.xs
            yArr = fish.ys
            vis = fish.vis
            for pointIdx in range(12):
                # Visibility is set to zero if they are out of bounds
                # Just got to clip them so that YOLO does not throw an error
                x = np.clip(xArr[pointIdx], 0, imageSizeX - 1)
                y = np.clip(yArr[pointIdx], 0, imageSizeY - 1)
                f.write(str(x / imageSizeX) + ' ' + str(y / imageSizeY)
                        + ' ' + str(int(vis[pointIdx])) + ' ')
            f.write('\n')

def save_image(frame_idx, im):
    biggestIdx4TrainingData = Config.biggestIdx4TrainingData
    dataDirectory = Config.dataDirectory

    subFolder = 'train/' if frame_idx < biggestIdx4TrainingData else 'val/'
    imagesPath = dataDirectory + '/' + 'images/' + subFolder
    strIdxInFormat = format(frame_idx, '06d')
    filename = 'zebrafish_' + strIdxInFormat + '.png'
    imagesPath += filename
    cv.imwrite(imagesPath, im)

def generate_and_save_data(frame_idx):
    # Calling the variables from the configuration file
    maxFishesInView = Config.maxFishesInView
    averageFishInEdges = Config.averageFishInEdges
    overlappingFishFrequency = Config.overlappingFishFrequency
    shouldAddStaticNoise = Config.shouldAddStaticNoise
    shouldAddPatchyNoise = Config.shouldAddStaticNoise
    shouldSaveAnnotations = Config.shouldSaveAnnotations
    shouldSaveImages = Config.shouldSaveImages

    fishesInView = np.random.randint(0, maxFishesInView)
    fishesInEdge = np.random.poisson(averageFishInEdges)
    overlappingFish = 0
    for _ in range(fishesInView + fishesInEdge):
        shouldItOverlap = True if np.random.rand() < overlappingFishFrequency else False
        if shouldItOverlap: overlappingFish += 1

    fishVectList, overlappingFishVectList = generateRandomConfiguration(fishesInView,fishesInEdge,overlappingFish)

    im, fish_list, overlap_fish_list = drawAquarium(fishVectList, overlappingFishVectList)

    if shouldAddStaticNoise:
        im = add_noise_static_noise(im)

    if shouldAddPatchyNoise:
        im = add_patchy_noise(im, fish_list, overlap_fish_list)

    if shouldSaveAnnotations:
        save_annotations(frame_idx, fish_list + overlap_fish_list)

    if shouldSaveImages:
        save_image(frame_idx, im)










