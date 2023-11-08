# This script contains all the functions required to render a synthetic larva, given the parameter vector of the physical model and the fish length

# Import libraries
import numpy as np
from numpy.random import normal as normrnd
from scipy.stats import norm
# import matplotlib.pyplot as plt
import cv2
import time
import pdb

######################
# Auxilliary functions
######################

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

# Sigmoid function
def sigmoid(x, scaling):
    y = 1 / (1 + np.exp(-scaling * x)) 
    return y

def custom_round(num):
    return np.floor(num) + np.round(num - np.floor(num) + 1) - 1


# Add Gaussian noise to image
def add_noise(noise_typ,image,mean,var):
   if noise_typ == "gauss":
      row,col= image.shape
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col,1)) * 255
      gauss[gauss < 0] = 0
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      noisy[noisy > 255] = 255
      return noisy


# Add mask to an image
def generate_mask(img):
    _, bw = cv2.threshold(np.uint8(img), 0, 255, cv2.THRESH_BINARY)
    # Choose dilation kernel whose size is randomly sampled from [3, 5, 7]
    kernel_size = np.random.randint(1,4) * 2 + 1
    kernel = np.ones((kernel_size, kernel_size), dtype = np.uint8)
    bw_dilated = cv2.dilate(bw, kernel, iterations = 1)
    return bw_dilated


#######################################
# Functions that directly render larva
#######################################

# Create an ellipsoid with given co-ordinates of the center and the dimensions of the axes
# This ellipsoid is used to construct eyes, head and belly of the larva
def drawEllipsoid(canvas, elpsd_ct, elpsd_a, elpsd_b, elpsd_c, brightness, theta, phi, gamma):
    x = np.linspace(0, canvas.shape[0] - 1, canvas.shape[0])
    y = np.linspace(0, canvas.shape[1] - 1, canvas.shape[1])
    z = np.linspace(0, canvas.shape[2] - 1, canvas.shape[2])
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='xy')
    # Co-ordinates of the ellipsoid shifted to its center
    XX = XX - elpsd_ct[0]
    YY = YY - elpsd_ct[1]
    ZZ = ZZ - elpsd_ct[2]
    # Reorient the ellipsoid based on input angles
    rot_mat = rotx(-gamma) @ roty(-phi) @ rotz(-theta)
    XX_transformed = rot_mat[0,0] * XX + rot_mat[0,1] * YY + rot_mat[0,2] * ZZ
    YY_transformed = rot_mat[1,0] * XX + rot_mat[1,1] * YY + rot_mat[1,2] * ZZ
    ZZ_transformed = rot_mat[2,0] * XX + rot_mat[2,1] * YY + rot_mat[2,2] * ZZ
    # Generate intensities of the model 
    model = 1 + (-(XX_transformed ** 2 / (2 * elpsd_a ** 2) + YY_transformed ** 2 / (2 * elpsd_b ** 2) + ZZ_transformed ** 2 / (2 * elpsd_c ** 2) - 1))
    model[model < 0] = 0
    return model



# Orthographic projection along the three camera axes directions (alias for axis direction is dimension below)
def project(model, dimension):
    vec = np.array([0, 1, 2])
    idx = np.argwhere(vec == dimension)
    mask = np.ones(len(vec), dtype = bool)
    mask[idx] = False
    vec = vec[mask]
    
    if dimension == 0:
        projection = np.squeeze(np.sum(model, axis = dimension))
        projection = projection.T
        projection = np.flip(projection, axis = 0)

    elif dimension == 1:
        projection = np.squeeze(np.sum(model, axis = dimension))
        projection = np.flip(projection.T, axis = 1)
        projection = np.flip(projection, axis = 0)

    elif dimension == 2:
        projection = np.squeeze(np.sum(model, axis = dimension))
    return projection



# Render 2-D projection of the larval anterior (top view) in a 3-D canvas of dimensions size_lut x size_lut x size_lut
def draw_anterior_b(seglen, theta, gamma, phi, dh1, dh2, dimension, size_lut, randomize):
    XX = size_lut
    YY = size_lut
    ZZ = size_lut
    
    # Distance between eyes
    d_eye = seglen * (randomize * normrnd(1, 0.05) + (1 - randomize))
    
    # Relative position of eyes, belly and head on the ball and stick model
    c_eyes = 1.9 * (randomize * normrnd(1, 0.05) + (1 - randomize))
    c_belly = 0.98 * (randomize * normrnd(1, 0.05) + (1 - randomize))
    c_head = 1.04 * (randomize * normrnd(1, 0.05) + (1 - randomize))
    canvas = np.zeros((XX, YY, ZZ))
    
    # Rotation matrix
    R = rotz(theta) @ roty(phi) @ rotx(gamma)
    
    # Initialize points of the ball and stick model in the canvas
    pt_original =  np.zeros((3, 3))
    # pt_original_1 is the mid-point in Python's indexing format
    pt_original[:,1] = np.array([np.floor(XX / 2) + dh1, np.floor(YY / 2) + dh2, np.floor(ZZ / 2)])
    pt_original[:,0] = pt_original[:,1] - np.array([seglen, 0, 0], dtype = pt_original.dtype) 
    pt_original[:,2] = pt_original[:,1] + np.array([seglen, 0, 0], dtype = pt_original.dtype) 
    
    # Initialize centers of eyes, belly and head with respect to the ball and stick model
    eye1_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]], 
        [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] + d_eye / 2],
        [pt_original[2,1] - seglen / 8]], dtype = pt_original.dtype)
    eye1_c = eye1_c - pt_original[:, 1, None]
    eye1_c = np.matmul(R, eye1_c) + pt_original[:, 1, None]
    
    eye2_c = np.array([[c_eyes * pt_original[0, 0] + (1 - c_eyes) * pt_original[0, 1]], 
        [c_eyes * pt_original[1, 0] + (1 - c_eyes) * pt_original[1, 1] - d_eye / 2],
        [pt_original[2,1] - seglen / 8]], dtype = pt_original.dtype)
    eye2_c = eye2_c - pt_original[:, 1, None]
    eye2_c = np.matmul(R, eye2_c) + pt_original[:, 1, None]

    belly_c = np.array([[c_belly * pt_original[0, 1] + (1 - c_belly) * pt_original[0, 2]],
            [c_belly * pt_original[1, 1] + (1 - c_belly) * pt_original[1, 2]], 
            [pt_original[2, 1] - seglen / 6]], dtype = pt_original.dtype)
    belly_c = belly_c - pt_original[:, 1, None]
    belly_c = np.matmul(R, belly_c) + pt_original[:, 1, None]

    head_c = np.array([[c_head * pt_original[0, 0] + (1 - c_head) * pt_original[0, 1]],
            [c_head * pt_original[1, 0] + (1 - c_head) * pt_original[1, 1]],
            [np.ceil(XX / 2) - seglen / 6]], dtype = pt_original.dtype)
    head_c = head_c - pt_original[:, 1, None]
    head_c = np.matmul(R, head_c) + pt_original[:, 1, None]
    # Set brightness of eyes, belly and head
    eyes_br = 235 * (randomize * normrnd(1, 0.1) + (1 - randomize))
    belly_br = eyes_br * 0.83 * (randomize * normrnd(1, 0.1) + (1 - randomize))
    head_br = belly_br * 0.64 * (randomize * normrnd(1, 0.1) + (1 - randomize))

    # Generate random variables for scaling sizes of eyes, head and belly
    rand1_eye = randomize * normrnd(1, 0.05) + (1 - randomize)
    rand2_eye = randomize * normrnd(1, 0.05) + (1 - randomize)
    rand3_eye = randomize * normrnd(1, 0.05) + (1 - randomize)
    rand1_belly = randomize * normrnd(1, 0.05) + (1 - randomize)
    rand2_belly = randomize * normrnd(1, 0.05) + (1 - randomize)
    rand1_head = randomize * normrnd(1, 0.05) + (1 - randomize)
    rand2_head = randomize * normrnd(1, 0.05) + (1 - randomize)
    
    # Set size of eyes, belly and head
    eye_w = seglen * 0.22 * rand1_eye
    eye_l = seglen * 0.35 * rand2_eye
    eye_h = seglen * 0.3
    belly_w = seglen * 0.29 * rand1_belly
    belly_l = seglen * 0.86 * rand2_belly
    belly_h = seglen * 0.34
    head_w = seglen * 0.3 * rand1_head
    head_l = seglen * 0.86 * rand2_head
    head_h = seglen * 0.53

    # Construct 3-D models of eyes, belly and head
    model_eye1 = drawEllipsoid(canvas, eye1_c, eye_l, eye_w, eye_h, eyes_br, theta + 0 * np.pi / 20 * rand3_eye, phi, gamma)
    model_eye2 = drawEllipsoid(canvas, eye2_c, eye_l, eye_w, eye_h, eyes_br, theta + 0 * np.pi / 20 * rand3_eye, phi, gamma)
    model_belly = drawEllipsoid(canvas, belly_c, belly_l, belly_w, belly_h, belly_br, theta, phi, gamma)
    model_head = drawEllipsoid(canvas, head_c, head_l, head_w, head_h, head_br, theta, phi, gamma)
    # Project eyes, belly and head models on independent 2-D images
    project_eye1 = project(model_eye1, dimension)
    project_eye1 = 2 * (sigmoid(project_eye1, 0.45) - 0.5) * eyes_br
    project_eye2 = project(model_eye2, dimension)
    project_eye2 = 2 * (sigmoid(project_eye2, 0.45) - 0.5) * eyes_br
    project_belly = project(model_belly, dimension)
    project_belly = 2 * (sigmoid(project_belly, 0.3) - 0.5) * belly_br
    project_head = project(model_head, dimension)
    project_head = 2 * (sigmoid(project_head, 0.4) - 0.5) * head_br
    
    # Blend eyes, belly and head 2-D projections into one image
    projection = np.uint8(np.maximum(np.maximum(np.maximum(project_eye1, project_eye2), project_belly), project_head))
    return projection, eye1_c, eye2_c, model_eye1, model_eye2, model_head, model_belly



# Render 2-D projection of the larval tail
def gen_lut_b_tail(n, seglen, d1, d2, t, randomize):
    size_lut = 29
    size_half = (size_lut + 1) / 2

    # Size of the balls in the ball and stick model
    random_number_size = randomize * normrnd(0.5, 0.1) + (1 - randomize) * 0.5
    ballsize = random_number_size * np.array([3, 2, 2, 2, 2, 1.5, 1.2, 1.2, 1])
    # Thickness of the sticks in the model
    thickness = random_number_size * np.array([7, 6, 5.5, 5, 4.5, 4, 3.5, 3])
    # Brightness of the tail
    b_tail = np.array([0.7, 0.55, 0.45, 0.40, 0.32, 0.28, 0.20, 0.15]) / 1.5

    imageSizeX = size_lut
    imageSizeY = size_lut

    columnsInImage0, rowsInImage0 = np.meshgrid(np.linspace(0, imageSizeX - 1 , imageSizeX), np.linspace(0, imageSizeY - 1, imageSizeY), indexing='xy')
    imblank = np.zeros((size_lut, size_lut), dtype = np.uint8)

    radius = ballsize[n + 1]
    th = thickness[n + 1]
    bt = b_tail[n]
    bt_gradient = b_tail[n + 1] / b_tail[n]
    p_max = norm.pdf(0, 0, th)
    centerX = (size_half - 1) + d1
    centerY = (size_half - 1) + d2
    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0
    ballpix = (rowsInImage - centerY) ** 2 + (columnsInImage - centerX) ** 2 <= radius ** 2
    ballpix = custom_round(custom_round(np.uint8(ballpix) * 255 * bt) * 0.85)
    pt = np.zeros((2,2))
    R = np.squeeze(np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]))
    vec = np.matmul(R, np.array([seglen, 0], dtype = np.float64))
    pt[:, 0] = np.array([(size_half - 1) + d1, (size_half - 1) + d2])
    pt[:, 1] = pt[:, 0] + vec
    stickpix = imblank
    columnsInImage = columnsInImage0
    rowsInImage = rowsInImage0 
    if (pt[0, 1] - pt[0, 0]) != 0:
        slope = (pt[1, 1] - pt[1, 0]) / (pt[0, 1] - pt[0, 0])
        # vectors perpendicular to the line segment
        # th is the thickness of the sticks in the model
        vp = np.array([-slope, 1]) / np.linalg.norm(np.array([-slope, 1]))
        # one vertex of the rectangle
        V1 = pt[:,1] - vp * th
        # two sides of the rectangle
        s1 = 2 * vp * th
        s2 = pt[:,0] - pt[:,1]
        # find the pixels inside the rectangle
        r1 = rowsInImage - V1[1]
        c1 = columnsInImage - V1[0]
        # innter products
        ip1 = r1 * s1[1] + c1 * s1[0]
        ip2 = r1 * s2[1] + c1 * s2[0]
        condition1_mask = np.zeros((ip1.shape[0], ip1.shape[1]), dtype = bool)
        condition1_mask[ip1 > 0] = True 
        condition2_mask = np.zeros((ip1.shape[0], ip1.shape[1]), dtype = bool)
        condition2_mask[ip1 < np.dot(s1, s1)] = True
        condition3_mask = np.zeros((ip2.shape[0], ip2.shape[1]), dtype = bool)
        condition3_mask[ip2 > 0] = True
        condition4_mask = np.zeros((ip2.shape[0], ip2.shape[1]), dtype = bool)
        condition4_mask[ip2 < np.dot(s2, s2)] = True
        stickpix_bw = np.logical_and.reduce((condition1_mask, condition2_mask, condition3_mask, condition4_mask))
    else:
        condition1_mask = np.zeros(rowsInImage.shape[0], rowsInImage.shape[1], dtype = bool)
        condition1_mask[rowsInImage < np.maximum(pt[1, 1], pt[1, 0])] = True
        condition2_mask = np.zeros(rowsInImage.shape[0], rowsInImage.shape[1], dtype = bool)
        condition2_mask[rowsInImage > np.minimum(pt[1, 1], pt[1, 0])] = True
        condition3_mask = np.zeros(columnsInImage.shape[0], columnsInImage.shape[1], dtype=bool)
        condition3_mask[columnsInImage < pt[0,1] + th] = True
        condition4_mask = np.zeros(columnsInImage.shape[0], columnsInImage.shape[1], dtype=bool)
        condition4_mask[columnsInImage > pt[0,1] - th] = True
        stickpix_bw = np.logical_and.reduce(condition1_mask, condition2_mask, condition3_mask, condition4_mask)
    
    # brightness of the points on the stick is a function of its distance to the segment
    idx_bw = np.argwhere(stickpix_bw == 1)
    ys = idx_bw[:,0]
    xs = idx_bw[:,1]
    px = pt[0, 1] - pt[0, 0]
    py = pt[1, 1] - pt[1, 0]
    pp = px * px + py * py
    # the distance between a pixel and the fish backbone
    d_radial = np.zeros((len(ys), 1))
    # the distance between a pixel and the anterior end of the segment (0 < d_axial < 1)
    b_axial = np.zeros((len(ys), 1))
    for i in range(0, len(ys)):
        u = ((xs[i] - pt[0,0]) * px + (ys[i] - pt[1,0]) * py) / pp
        dx = pt[0, 0] + u * px - xs[i]
        dy = pt[1,0] + u * py - ys[i]
        d_radial[i] = dx * dx + dy * dy
        b_axial[i] = 1 - (1 - bt_gradient) * u * 0.9
    b_stick = np.uint8(255 * (norm.pdf(d_radial, 0, th) / p_max))
    for i in range(0, len(ys)):
        stickpix[ys[i], xs[i]] = custom_round(b_stick[i] * b_axial[i])
    stickpix = custom_round(stickpix * bt)
    graymodel = np.maximum(ballpix, stickpix)
    
    return graymodel



# Convert parameters to a grayscale image
# Returns grayscale image and the corresponding annotations of the 2-D pose
def f_x_to_model(x, seglen, randomize):
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

    # Construct headpix (larval anterior)
    size_lut = 49
    size_half = (size_lut + 1) / 2

    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])
    fish_anterior, eye1_c, eye2_c, _, _, _, _ = draw_anterior_b(seglen, x[2], 0, 0, dh1, dh2, 2, size_lut, randomize)
    min_Y = pt[1, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis = 1) > 0)[0]
    max_Y = pt[1, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis = 1) > 0)[-1] + 1
    min_X = pt[0, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis = 0) > 0)[0]
    max_X = pt[0, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis = 0) > 0)[-1] + 1 

    new_origin = np.array([[np.minimum(np.min(pt[0, :]), min_X)], [np.minimum(np.min(pt[1, :]), min_Y)]])
    new_origin[0] = new_origin[0] - (np.random.rand(1) * randomize + (1 - randomize))* np.minimum(new_origin[0], 30) # shift the fish by a random distance along x - axis
    new_origin[1] = new_origin[1] - (np.random.rand(1) * randomize + (1 - randomize)) * np.minimum(new_origin[1], 30) # shift the fish by a random distance along y - axis
    new_origin[0] = 0
    new_origin[1] = 0

    imageSizeX = np.maximum(np.uint16(np.max(pt[0, :])), max_X) - new_origin[0] 
    imageSizeY = np.maximum(np.uint16(np.max(pt[1, :])), max_Y) - new_origin[1]
    
    imageSizeX = np.single(imageSizeX + (np.random.rand(1) * randomize + (1 - randomize)) * np.minimum(101 - imageSizeX, 30)) 
    imageSizeY = np.single(imageSizeY + (np.random.rand(1) * randomize + (1 - randomize)) * np.minimum(101 - imageSizeY, 30)) 
    imageSizeX = 648
    imageSizeY = 488 
    
    pt[0, :] = pt[0, :] - new_origin[0]
    pt[1, :] = pt[1, :] - new_origin[1]
    eye1_c[0] = eye1_c[0] - (size_half - 1) + pt[0, 1]
    eye1_c[1] = eye1_c[1] - (size_half - 1)+ pt[1, 1]
    eye2_c[0] = eye2_c[0] - (size_half - 1)+ pt[0, 1]
    eye2_c[1] = eye2_c[1] - (size_half - 1)+ pt[1, 1]
    pt = np.concatenate([pt, eye1_c[0 : 2], eye2_c[0 : 2]], axis=1)
    imblank = np.zeros((int(imageSizeY), int(imageSizeX)), dtype=np.uint8)
    headpix = imblank.copy()
    bodypix = imblank.copy()
    coor_h1 = np.floor(pt[0, 1])
    coor_h2 = np.floor(pt[1, 1])
    headpix[int(np.maximum(0, coor_h2 - (size_half - 1))) : int(np.minimum((imageSizeY), 1 + coor_h2 + (size_half - 1))), 
            int(np.maximum(0, coor_h1 - (size_half - 1))) : int(np.minimum((imageSizeX), 1 + coor_h1 + (size_half - 1)))] = fish_anterior[int(np.maximum((size_half - 1) - coor_h2, 0)) : int(np.minimum((imageSizeY) - coor_h2 + size_half - 1, size_lut)),
                    int(np.maximum((size_half - 1) - coor_h1, 0)) : int(np.minimum((imageSizeX) - coor_h1 + size_half - 1, size_lut))]
    
    # Construct tailpix (larval posterior)
    size_lut = 29
    size_half = (size_lut + 1) / 2
    for ni in range(0,7):
        n = ni + 2
        coor_t1 = np.floor(pt[0, n])
        dt1 = pt[0, n] - coor_t1
        coor_t2 = np.floor(pt[1, n])
        dt2 = pt[1, n] - coor_t2
        tailpix = imblank
        tail_model = gen_lut_b_tail(ni, seglen, dt1, dt2, theta[n], randomize)
        tailpix[int(np.maximum(0, coor_t2 - (size_half - 1))) : int(np.minimum(imageSizeY, 1 + coor_t2 + (size_half - 1))),
                int(np.maximum(0, coor_t1 - (size_half - 1))) : int(np.minimum(imageSizeX, 1 + coor_t1 + (size_half - 1)))] = tail_model[int(np.maximum((size_half - 1) - coor_t2, 0)) : int(np.minimum(imageSizeY - coor_t2 + size_half - 1, size_lut)),
                    int(np.maximum((size_half - 1) - coor_t1, 0)) : int(np.minimum(imageSizeX - coor_t1 + size_half - 1, size_lut))]
        bodypix = np.maximum(bodypix, tailpix)
    
    # Combine headpix and tailpix into a single image
    graymodel = np.maximum(headpix, normrnd(1, 0.1) * bodypix)
    return graymodel, pt, new_origin




############################################################################################################################################################
# Convert parameters to a grayscale image
# Returns grayscale image and the corresponding annotations of the 2-D pose
# USE THIS FOR POSE EVALUATION
def f_x_to_model_evaluation(x, seglen, randomize, imageSizeX, imageSizeY):
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

    # Construct headpix (larval anterior)
    size_lut = 49
    size_half = (size_lut + 1) / 2
    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])
    fish_anterior, eye1_c, eye2_c, _, _, _, _ = draw_anterior_b(seglen, x[2], 0, 0, dh1, dh2, 2, size_lut, randomize)
    imblank = np.zeros((int(imageSizeY), int(imageSizeX)), dtype=np.uint8)
    headpix = imblank.copy()
    bodypix = imblank.copy()
    coor_h1 = np.floor(pt[0, 1])
    coor_h2 = np.floor(pt[1, 1])
    headpix[int(np.maximum(0, coor_h2 - (size_half - 1))) : int(np.minimum((imageSizeY), 1 + coor_h2 + (size_half - 1))),
            int(np.maximum(0, coor_h1 - (size_half - 1))) : int(np.minimum((imageSizeX), 1 + coor_h1 + (size_half - 1)))] = fish_anterior[int(np.maximum((size_half - 1) - coor_h2, 0)) : int(np.minimum((imageSizeY) - coor_h2 + size_half - 1, size_lut)),
                    int(np.maximum((size_half - 1) - coor_h1, 0)) : int(np.minimum((imageSizeX) - coor_h1 + size_half - 1, size_lut))]
    
    # Construct tailpix (larval posterior)
    size_lut = 29
    size_half = (size_lut + 1) / 2
    for ni in range(0,7):
        n = ni + 2
        coor_t1 = np.floor(pt[0, n])
        dt1 = pt[0, n] - coor_t1
        coor_t2 = np.floor(pt[1, n])
        dt2 = pt[1, n] - coor_t2
        tailpix = imblank
        tail_model = gen_lut_b_tail(ni, seglen, dt1, dt2, theta[n], randomize)
        tailpix[int(np.maximum(0, coor_t2 - (size_half - 1))) : int(np.minimum(imageSizeY, 1 + coor_t2 + (size_half - 1))),
                int(np.maximum(0, coor_t1 - (size_half - 1))) : int(np.minimum(imageSizeX, 1 + coor_t1 + (size_half - 1)))] = tail_model[int(np.maximum((size_half - 1) - coor_t2, 0)) : int(np.minimum(imageSizeY - coor_t2 + size_half - 1, size_lut)),
                    int(np.maximum((size_half - 1) - coor_t1, 0)) : int(np.minimum(imageSizeX - coor_t1 + size_half - 1, size_lut))]
        bodypix = np.maximum(bodypix, tailpix)

    # Combine headpix and tailpix into a single image
    graymodel = np.maximum(headpix, normrnd(1, 0.1) * bodypix)
    return graymodel, pt

#####################################
# NOTE: What I added 
#
# def f_x_model_bigger(x, seglen, randomize, imageSizeX, imageSizeY):
#     bCanvas = np.zeros((imageSizeY, imageSizeX))
#     graymodel, pt, origin = f_x_to_model(x, seglen, randomize)
#     np.save('anns_og.npy', pt)
#     np.save('grays_og.npy', graymodel)
#
#     origin = np.squeeze(origin)
#     pdb.set_trace()
#     pt[0,:] += origin[0]
#     pt[1,:] += origin[1]
#
#     size = graymodel.shape
#     crop_b = np.array([origin[1], origin[1] + size[0] - 1, origin[0], origin[0] + size[1] - 1 ])
#     imageSizeYForSmallerPicture, imageSizeXForSmallerPicture = size
#
#     smallY_b, bigY_b, smallX_b, bigX_b = crop_b
#     canvas_b_indices = np.copy(crop_b)
#     canvas_b_indices = canvas_b_indices.astype(int)
#     gray_b_indices = np.array([0, imageSizeYForSmallerPicture - 1, 0, imageSizeXForSmallerPicture - 1])
#     gray_b_indices = gray_b_indices.astype(int)
#     gray_b = graymodel
#
#     # Adjusting if gray_b is horizontally out of bounds
#     isGrayBCompletelyOutOfBounds = False
#     if smallX_b < 0:
#         # Atleast some part of the image is out of bounds on the left
#         if bigX_b < 0:
#             isGrayBCompletelyOutOfBounds = True
#         else:
#             canvas_b_indices[2] = 0
#             lenghtOfImageNotShowing = -smallX_b
#             gray_b_indices[2] = lenghtOfImageNotShowing
#     if bigX_b > imageSizeX - 1:
#         # At least some part of the image is out of bounds on the right
#         if smallX_b > imageSizeX - 1:
#             isGrayBCompletelyOutOfBounds = True
#         else:
#             canvas_b_indices[3] = imageSizeX - 1
#             lenghtOfImageShowing = imageSizeX - smallX_b
#             gray_b_indices[3] = lenghtOfImageShowing - 1
#     # Adjusting if gray_b is vertically out of bounds
#     if smallY_b < 0:
#         # Atleast some part of the image is out of bounds on the left
#         if bigY_b < 0:
#             isGrayBCompletelyOutOfBounds = True
#         else:
#             canvas_b_indices[0] = 0
#             lenghtOfImageNotShowing = -smallY_b
#             gray_b_indices[0] = lenghtOfImageNotShowing
#     if bigY_b > imageSizeY - 1:
#         # At least some part of the image is out of bounds on the right
#         if smallY_b > imageSizeY - 1:
#             isGrayBCompletelyOutOfBounds = True
#         else:
#             canvas_b_indices[1] = imageSizeY - 1
#             lenghtOfImageShowing = imageSizeY - smallY_b
#             gray_b_indices[1] = lenghtOfImageShowing - 1
#
#     print('canvas_b_indices: ',canvas_b_indices)
#     print('gray_b_indices: ',gray_b_indices)
#     if not isGrayBCompletelyOutOfBounds:
#         bCanvas[ canvas_b_indices[0]: canvas_b_indices[1] + 1, canvas_b_indices[2]: canvas_b_indices[3] + 1] = \
#             gray_b[gray_b_indices[0]: gray_b_indices[1] + 1, gray_b_indices[2]: gray_b_indices[3] + 1]
#     bCanvas = bCanvas.astype(np.uint8)
#
#     return bCanvas, pt

# Making a function that always draws a fish in the center to allow us to draw fishes on the edges without crashing
def f_x_to_model_centered(x, seglen, randomize):
    smallImageSizeX, smallImageSizeY = 141, 141

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

    # Construct headpix (larval anterior)
    size_lut = 49
    size_half = (size_lut + 1) / 2

    dh1 = pt[0, 1] - np.floor(pt[0, 1])
    dh2 = pt[1, 1] - np.floor(pt[1, 1])
    fish_anterior, eye1_c, eye2_c, _, _, _, _ = draw_anterior_b(seglen, x[2], 0, 0, dh1, dh2, 2, size_lut, randomize)
    min_Y = pt[1, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis=1) > 0)[0]
    max_Y = pt[1, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis=1) > 0)[-1] + 1
    min_X = pt[0, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis=0) > 0)[0]
    max_X = pt[0, 1] - (size_half) + np.argwhere(np.sum(fish_anterior, axis=0) > 0)[-1] + 1

    new_origin = np.array([[np.minimum(np.min(pt[0, :]), min_X)], [np.minimum(np.min(pt[1, :]), min_Y)]])
    new_origin[0] = new_origin[0] - (np.random.rand(1) * randomize + (1 - randomize)) * np.minimum(new_origin[0],
                                                                                                   30)  # shift the fish by a random distance along x - axis
    new_origin[1] = new_origin[1] - (np.random.rand(1) * randomize + (1 - randomize)) * np.minimum(new_origin[1],
                                                                                                   30)  # shift the fish by a random distance along y - axis
    new_origin[0] = 0
    new_origin[1] = 0
    # NOTE: I should probably round off until the end
    new_origin[0] = custom_round( pt[0,0]  ) - custom_round( smallImageSizeX / 2 )
    new_origin[1] = custom_round( pt[1, 0] ) - custom_round( smallImageSizeY / 2)


    imageSizeX = np.maximum(np.uint16(np.max(pt[0, :])), max_X) - new_origin[0]
    imageSizeY = np.maximum(np.uint16(np.max(pt[1, :])), max_Y) - new_origin[1]
    imageSizeX = smallImageSizeX
    imageSizeY = smallImageSizeY


    imageSizeX = np.single(
        imageSizeX + (np.random.rand(1) * randomize + (1 - randomize)) * np.minimum(101 - imageSizeX, 30))
    imageSizeY = np.single(
        imageSizeY + (np.random.rand(1) * randomize + (1 - randomize)) * np.minimum(101 - imageSizeY, 30))
    imageSizeX = smallImageSizeX
    imageSizeY = smallImageSizeY

    pt[0, :] = pt[0, :] - new_origin[0]
    pt[1, :] = pt[1, :] - new_origin[1]
    eye1_c[0] = eye1_c[0] - (size_half - 1) + pt[0, 1]
    eye1_c[1] = eye1_c[1] - (size_half - 1) + pt[1, 1]
    eye2_c[0] = eye2_c[0] - (size_half - 1) + pt[0, 1]
    eye2_c[1] = eye2_c[1] - (size_half - 1) + pt[1, 1]
    pt = np.concatenate([pt, eye1_c[0: 2], eye2_c[0: 2]], axis=1)
    imblank = np.zeros((int(imageSizeY), int(imageSizeX)), dtype=np.uint8)
    headpix = imblank.copy()
    bodypix = imblank.copy()
    coor_h1 = np.floor(pt[0, 1])
    coor_h2 = np.floor(pt[1, 1])
    headpix[int(np.maximum(0, coor_h2 - (size_half - 1))): int(np.minimum((imageSizeY), 1 + coor_h2 + (size_half - 1))),
    int(np.maximum(0, coor_h1 - (size_half - 1))): int(
        np.minimum((imageSizeX), 1 + coor_h1 + (size_half - 1)))] = fish_anterior[
                                                                    int(np.maximum((size_half - 1) - coor_h2, 0)): int(
                                                                        np.minimum(
                                                                            (imageSizeY) - coor_h2 + size_half - 1,
                                                                            size_lut)),
                                                                    int(np.maximum((size_half - 1) - coor_h1, 0)): int(
                                                                        np.minimum(
                                                                            (imageSizeX) - coor_h1 + size_half - 1,
                                                                            size_lut))]

    # Construct tailpix (larval posterior)
    size_lut = 29
    size_half = (size_lut + 1) / 2
    for ni in range(0, 7):
        n = ni + 2
        coor_t1 = np.floor(pt[0, n])
        dt1 = pt[0, n] - coor_t1
        coor_t2 = np.floor(pt[1, n])
        dt2 = pt[1, n] - coor_t2
        tailpix = imblank
        tail_model = gen_lut_b_tail(ni, seglen, dt1, dt2, theta[n], randomize)
        tailpix[
        int(np.maximum(0, coor_t2 - (size_half - 1))): int(np.minimum(imageSizeY, 1 + coor_t2 + (size_half - 1))),
        int(np.maximum(0, coor_t1 - (size_half - 1))): int(
            np.minimum(imageSizeX, 1 + coor_t1 + (size_half - 1)))] = tail_model[
                                                                      int(np.maximum((size_half - 1) - coor_t2,
                                                                                     0)): int(np.minimum(
                                                                          imageSizeY - coor_t2 + size_half - 1,
                                                                          size_lut)),
                                                                      int(np.maximum((size_half - 1) - coor_t1,
                                                                                     0)): int(np.minimum(
                                                                          imageSizeX - coor_t1 + size_half - 1,
                                                                          size_lut))]
        bodypix = np.maximum(bodypix, tailpix)

    # Combine headpix and tailpix into a single image
    graymodel = np.maximum(headpix, normrnd(1, 0.1) * bodypix)
    return graymodel, pt, new_origin

def f_x_to_model_bigger(x, seglen, randomize, imageSizeX = 648, imageSizeY = 488):
    """
        Essentially a wrapper to send centered images to a bigger canvas in order to avoid out of bounds errors
        as well as speed up the program since it does not need to draw the fish parts onto a bigger canvas
    :param x: x numpy vector of size 11
    :param seglen: float segment lenght
    :param randomize: 1 or 0, serves as a flag to randomize the fish
    :return: numpy grayscale image, numpy array representing the annotations of the fish
    """
    graymodel, pt, new_origin = f_x_to_model_centered(x, seglen, randomize)
    canvas = np.zeros((imageSizeY, imageSizeX))
    new_origin = np.array([ new_origin[0][0][0], new_origin[1][0][0] ])
    pt[0,:] = pt[0,:] + new_origin[0]
    pt[1,:] = pt[1,:] + new_origin[1]

    size = graymodel.shape
    smallImageSizeY, smallImageSizeX = size[0], size[1]
    ###############
    # drawing the small gray scale images onto the bigger arrays
    bCanvas = np.zeros((imageSizeY, imageSizeX))
    crop_b = np.array([ new_origin[1], new_origin[1] + smallImageSizeY - 1,
                        new_origin[0], new_origin[0] + smallImageSizeX - 1])
    crop_b = crop_b.astype(int)

    smallY_b, bigY_b, smallX_b, bigX_b = crop_b
    canvas_b_indices = np.copy(crop_b)
    canvas_b_indices = canvas_b_indices.astype(int)
    gray_b_indices = np.array([0, smallImageSizeY - 1, 0, smallImageSizeX - 1])
    gray_b_indices = gray_b_indices.astype(int)

    # Adjusting if gray_b is horizontally out of bounds
    isGrayBCompletelyOutOfBounds = False
    if smallX_b < 0:
        # Atleast some part of the image is out of bounds on the left
        if bigX_b < 0:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_b_indices[2] = 0
            lenghtOfImageNotShowing = -smallX_b
            gray_b_indices[2] = lenghtOfImageNotShowing
    if bigX_b > imageSizeX - 1:
        # At least some part of the image is out of bounds on the right
        if smallX_b > imageSizeX - 1:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_b_indices[3] = imageSizeX - 1
            lenghtOfImageShowing = imageSizeX - smallX_b
            gray_b_indices[3] = lenghtOfImageShowing - 1

    # Adjusting if gray_b is vertically out of bounds
    if smallY_b < 0:
        # Atleast some part of the image is out of bounds on the left
        if bigY_b < 0:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_b_indices[0] = 0
            lenghtOfImageNotShowing = -smallY_b
            gray_b_indices[0] = lenghtOfImageNotShowing
    if bigY_b > imageSizeY - 1:
        # At least some part of the image is out of bounds on the right
        if smallY_b > imageSizeY - 1:
            isGrayBCompletelyOutOfBounds = True
        else:
            canvas_b_indices[1] = imageSizeY - 1
            lenghtOfImageShowing = imageSizeY - smallY_b
            gray_b_indices[1] = lenghtOfImageShowing - 1

    if not isGrayBCompletelyOutOfBounds:
        bCanvas[ canvas_b_indices[0]: canvas_b_indices[1] + 1, canvas_b_indices[2]: canvas_b_indices[3] + 1] = \
            graymodel[gray_b_indices[0]: gray_b_indices[1] + 1, gray_b_indices[2]: gray_b_indices[3] + 1]

    bCanvas = bCanvas.astype(np.uint8)

    return bCanvas, pt





