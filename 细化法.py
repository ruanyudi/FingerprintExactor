import cv2
import numpy as np
from tkinter import *
from math import *
from tkinter.filedialog import *

import numpy as np
import scipy
from PIL import ImageTk, Image
import cv2
from scipy import ndimage
from scipy import signal

def frequest(im, orientim, windsze, minWaveLength, maxWaveLength):
    rows, cols = np.shape(im)

     
     
     

    cosorient = np.mean(np.cos(2 * orientim))
    sinorient = np.mean(np.sin(2 * orientim))
    orient = atan2(sinorient, cosorient) / 2

     

     
     
    rotim = scipy.ndimage.rotate(im, orient / np.pi * 180 + 90, axes=(1, 0), reshape=False, order=3, mode='nearest')

     
     
     

    cropsze = int(np.fix(rows / np.sqrt(2)))
    offset = int(np.fix((rows - cropsze) / 2))
    rotim = rotim[offset:offset + cropsze][:, offset:offset + cropsze]

     
     

    proj = np.sum(rotim, axis=0)
    dilation = scipy.ndimage.grey_dilation(proj, windsze, structure=np.ones(windsze))

    temp = np.abs(dilation - proj)

    peak_thresh = 2

    maxpts = (temp < peak_thresh) & (proj > np.mean(proj))
    maxind = np.where(maxpts)

    rows_maxind, cols_maxind = np.shape(maxind)

     
     
     
     

    if cols_maxind < 2:
        freqim = np.zeros(im.shape)
    else:
        NoOfPeaks = cols_maxind
        waveLength = (maxind[0][cols_maxind - 1] - maxind[0][0]) / (NoOfPeaks - 1)
        if waveLength >= minWaveLength and waveLength <= maxWaveLength:
            freqim = 1 / np.double(waveLength) * np.ones(im.shape)
        else:
            freqim = np.zeros(im.shape)

    return freqim

def Normalization(img):
    mean = np.mean(img)
    std = np.std(img)
    # for row in range(img.shape[0]):
    #     for col in range(img.shape[1]):
    #         if img[row,col] >= mean:
    #             img[row,col] = 140+(60/std*(img[row,col]-mean)**2)**0.5
    #         else:
    #             img[row,col] = 140-(60/std*(img[row,col]-mean)**2)**0.5
    return (img-mean)/std
     
     
     
     
     
     
def ridge_segment(im, blksze, thresh):   

    rows, cols = im.shape

    im = Normalization(im)   
     

    new_rows = np.int32(blksze * np.ceil((np.float32(rows)) / (np.float32(blksze))))
    new_cols = np.int32(blksze * np.ceil((np.float32(cols)) / (np.float32(blksze))))

    padded_img = np.zeros((new_rows, new_cols))
    stddevim = np.zeros((new_rows, new_cols))

    padded_img[0:rows][:, 0:cols] = im

    for i in range(0, new_rows, blksze):
        for j in range(0, new_cols, blksze):
            block = padded_img[i:i + blksze][:, j:j + blksze]

            stddevim[i:i + blksze][:, j:j + blksze] = np.std(block) * np.ones(block.shape)

    stddevim = stddevim[0:rows][:, 0:cols]

    mask = stddevim > thresh

    mean_val = np.mean(im[mask])

    std_val = np.std(im[mask])

    normim = (im - mean_val) / (std_val)
     

    return (normim, mask)

def ridge_orient(im, gradientsigma, blocksigma, orientsmoothsigma):
    rows, cols = im.shape
     
    sze = np.fix(6 * gradientsigma)
    if np.remainder(sze, 2) == 0:
        sze = sze + 1

    gauss = cv2.getGaussianKernel(np.int32(sze), gradientsigma)
    f = gauss * gauss.T

    fy, fx = np.gradient(f)   

     
     

    Gx = signal.convolve2d(im, fx, mode='same')
    Gy = signal.convolve2d(im, fy, mode='same')

    Gxx = np.power(Gx, 2)
    Gyy = np.power(Gy, 2)
    Gxy = Gx * Gy

     

    sze = np.fix(6 * blocksigma)

    gauss = cv2.getGaussianKernel(np.int32(sze), blocksigma)
    f = gauss * gauss.T

    Gxx = ndimage.convolve(Gxx, f)
    Gyy = ndimage.convolve(Gyy, f)
    Gxy = 2 * ndimage.convolve(Gxy, f)

     
    denom = np.sqrt(np.power(Gxy, 2) + np.power((Gxx - Gyy), 2)) + np.finfo(float).eps

    sin2theta = Gxy / denom   
    cos2theta = (Gxx - Gyy) / denom

    if orientsmoothsigma:
        sze = np.fix(6 * orientsmoothsigma)
        if np.remainder(sze, 2) == 0:
            sze = sze + 1
        gauss = cv2.getGaussianKernel(np.int32(sze), orientsmoothsigma)
        f = gauss * gauss.T
        cos2theta = ndimage.convolve(cos2theta, f)   
        sin2theta = ndimage.convolve(sin2theta, f)   

    orientim = np.pi / 2 + np.arctan2(sin2theta, cos2theta) / 2
    return orientim



def image_enhance():
    global img
    blksze = 8
    thresh = 0.3
    normim, mask = ridge_segment(img, blksze, thresh)   
    cv2.imshow("norm", normim)
    mask_show = np.zeros(img.shape)
    mask_show[mask] = 1
    cv2.imshow('mask',mask_show)

    gradientsigma = 1
    blocksigma = 7
    orientsmoothsigma = 7
    orientim = ridge_orient(normim, gradientsigma, blocksigma, orientsmoothsigma)   
    cv2.imshow("orient", orientim)

    blksze = 16
    windsze = 5
    minWaveLength = 5
    maxWaveLength = 15
    freq, medfreq = ridge_freq(normim, mask, orientim, blksze, windsze, minWaveLength,
                               maxWaveLength)   
    cv2.imshow("freq", freq)

    freq = medfreq * mask
    kx = 0.65
    ky = 0.65
    newim = ridge_filter(normim, orientim, freq, kx, ky)   
    cv2.imshow("new",newim)

    img = 255 * (newim >= -3)

def ridge_filter(im, orient, freq, kx, ky):
    angleInc = 3
    im = np.double(im)
    rows, cols = im.shape
    newim = np.zeros((rows, cols))

    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

     
     

    non_zero_elems_in_freq = freq_1d[0][ind]
    non_zero_elems_in_freq = np.double(np.round((non_zero_elems_in_freq * 100))) / 100

    unfreq = np.unique(non_zero_elems_in_freq)

     
     

    sigmax = 1 / unfreq[0] * kx
    sigmay = 1 / unfreq[0] * ky

    sze = np.int32(np.round(3 * np.max([sigmax, sigmay])))

    x, y = np.meshgrid(np.linspace(-sze, sze, (2 * sze + 1)), np.linspace(-sze, sze, (2 * sze + 1)))

    reffilter = np.exp(-(((np.power(x, 2)) / (sigmax * sigmax) + (np.power(y, 2)) / (sigmay * sigmay)))) * np.cos(
        2 * np.pi * unfreq[0] * x)   

    filt_rows, filt_cols = reffilter.shape

    angleRange = np.int32(180 / angleInc)

    gabor_filter = np.array(np.zeros((angleRange, filt_rows, filt_cols)))

    for o in range(0, angleRange):
         
         
         

        rot_filt = scipy.ndimage.rotate(reffilter, -(o * angleInc + 90), reshape=False)
        gabor_filter[o] = rot_filt

     
     

    maxsze = int(sze)

    temp = freq > 0
    validr, validc = np.where(temp)

    temp1 = validr > maxsze
    temp2 = validr < rows - maxsze
    temp3 = validc > maxsze
    temp4 = validc < cols - maxsze

    final_temp = temp1 & temp2 & temp3 & temp4

    finalind = np.where(final_temp)

     
     

    maxorientindex = np.round(180 / angleInc)
    orientindex = np.round(orient / np.pi * 180 / angleInc)

     

    for i in range(0, rows):
        for j in range(0, cols):
            if orientindex[i][j] < 1:
                orientindex[i][j] = orientindex[i][j] + maxorientindex
            if orientindex[i][j] > maxorientindex:
                orientindex[i][j] = orientindex[i][j] - maxorientindex
    finalind_rows, finalind_cols = np.shape(finalind)
    sze = int(sze)
    for k in range(0, finalind_cols):
        r = validr[finalind[0][k]]
        c = validc[finalind[0][k]]

        img_block = im[r - sze:r + sze + 1][:, c - sze:c + sze + 1]

        newim[r][c] = np.sum(img_block * gabor_filter[int(orientindex[r][c]) - 1])

    return newim

def VThin(image, array):
     
     
    h, w = image.shape
    NEXT = 1
    for i in range(h):
        for j in range(w):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i, j - 1] + image[i, j] + image[i, j + 1] if 0 < j < w - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image

def feature():
    image_enhance()
    thinning()
    global img
    features = []
    h, w = img.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if img[i, j] == 0:  # 像素点为黑
                m = i
                n = j

                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]

                if sum(eightField) / 255 == 7:  # 黑色块1个，端点

                    # 判断是否为指纹图像边缘
                    if sum(img[:i, j]) == 255 * i or sum(img[i + 1:, j]) == 255 * (w - i - 1) or sum(
                            img[i, :j]) == 255 * j or sum(img[i, j + 1:]) == 255 * (h - j - 1):
                        continue
                    canContinue = TRUE
                    # print(m, n)
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            index = o
                            m = coordinate[o][0]
                            n = coordinate[o][1]
                            # print(m, n, index)
                            break
                    # print(m, n, index)
                    for k in range(4):
                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1], img[m, n + 1],
                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                        if sum(eightField) / 255 == 6:  # 连接点
                            for o in range(8):
                                if eightField[o] == 0 and o != 7 - index:
                                    index = o
                                    m = coordinate[o][0]
                                    n = coordinate[o][1]
                                    # print(m, n, index)
                                    break
                        else:
                            # print("false", i, j)
                            canContinue = FALSE
                    if canContinue:

                        if n - j != 0:
                            if i - m >= 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) + pi
                            elif i - m < 0 and j - n > 0:
                                direction = atan((i - m) / (n - j)) - pi
                            else:
                                direction = atan((i - m) / (n - j))
                        else:
                            if i - m >= 0:
                                direction = pi / 2
                            else:
                                direction = -pi / 2
                        feature = []
                        feature.append(i)
                        feature.append(j)
                        feature.append("endpoint")
                        feature.append(direction)
                        features.append(feature)

                elif sum(eightField) / 255 == 5:  # 黑色块3个，分叉点
                    coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1], [m + 1, n - 1],
                                  [m + 1, n], [m + 1, n + 1]]
                    junctionCoordinates = []
                    junctions = []
                    canContinue = TRUE
                    # 筛除不符合的分叉点
                    for o in range(8):  # 寻找相连接的下一个点
                        if eightField[o] == 0:
                            junctions.append(o)
                            junctionCoordinates.append(coordinate[o])
                    for k in range(3):
                        if k == 0:
                            a = junctions[0]
                            b = junctions[1]
                        elif k == 1:
                            a = junctions[1]
                            b = junctions[2]
                        else:
                            a = junctions[0]
                            b = junctions[2]
                        if (a == 0 and b == 1) or (a == 1 and b == 2) or (a == 2 and b == 4) or (a == 4 and b == 7) or (
                                a == 6 and b == 7) or (a == 5 and b == 6) or (a == 3 and b == 5) or (a == 0 and b == 3):
                            canContinue = FALSE
                            break

                    if canContinue:  # 合格分叉点
                        # print(junctions)
                        print(junctionCoordinates)
                        print(i, j, "合格分叉点")
                        directions = []
                        canContinue = TRUE
                        for k in range(3):  # 分三路进行
                            if canContinue:
                                junctionCoordinate = junctionCoordinates[k]
                                m = junctionCoordinate[0]
                                n = junctionCoordinate[1]
                                print(m, n, "start")
                                eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1], img[m, n - 1],
                                              img[m, n + 1],
                                              img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1], [m, n + 1],
                                              [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                canContinue = FALSE
                                for o in range(8):
                                    if eightField[o] == 0:
                                        a = coordinate[o][0]
                                        b = coordinate[o][1]
                                        print("a=", a, "b=", b)
                                        # print("i=", i, "j=", j)
                                        if (a != i or b != j) and (
                                                a != junctionCoordinates[0][0] or b != junctionCoordinates[0][1]) and (
                                                a != junctionCoordinates[1][0] or b != junctionCoordinates[1][1]) and (
                                                a != junctionCoordinates[2][0] or b != junctionCoordinates[2][1]):
                                            index = o
                                            m = a
                                            n = b
                                            canContinue = TRUE
                                            print(m, n, index, "支路", k)
                                            break
                                if canContinue:  # 能够找到第二个支路点
                                    for p in range(3):
                                        coordinate = [[m - 1, n - 1], [m - 1, n], [m - 1, n + 1], [m, n - 1],
                                                      [m, n + 1],
                                                      [m + 1, n - 1], [m + 1, n], [m + 1, n + 1]]
                                        eightField = [img[m - 1, n - 1], img[m - 1, n], img[m - 1, n + 1],
                                                      img[m, n - 1],
                                                      img[m, n + 1],
                                                      img[m + 1, n - 1], img[m + 1, n], img[m + 1, n + 1]]
                                        if sum(eightField) / 255 == 6:  # 连接点
                                            for o in range(8):
                                                if eightField[o] == 0 and o != 7 - index:
                                                    index = o
                                                    m = coordinate[o][0]
                                                    n = coordinate[o][1]
                                                    print(m, n, index, "支路尾")
                                                    # print(m, n, index)
                                                    break
                                        else:
                                            # print("false", i, j)
                                            canContinue = FALSE
                                if canContinue:  # 能够找到3个连接点

                                    if n - j != 0:
                                        if i - m >= 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) + pi
                                        elif i - m < 0 and j - n > 0:
                                            direction = atan((i - m) / (n - j)) - pi
                                        else:
                                            direction = atan((i - m) / (n - j))
                                    else:
                                        if i - m >= 0:
                                            direction = pi / 2
                                        else:
                                            direction = -pi / 2
                                    # print(direction)
                                    directions.append(direction)
                        if canContinue:
                            feature = []
                            feature.append(i)
                            feature.append(j)
                            feature.append("bifurcation")
                            feature.append(directions)
                            features.append(feature)
    print(features)
    for m in range(len(features)):
        if features[m][2] == "endpoint":
            cv2.circle(img, (features[m][1], features[m][0]), 3, (0, 0, 255), 1)
        else:
            cv2.circle(img, (features[m][1], features[m][0]), 3, (0, 0, 255), -1)
    global imgCopy
    cv2.imshow('Feature',img*1.0)

def HThin(image, array):
     
     
    h, w = image.shape
    NEXT = 1
    for j in range(w):
        for i in range(h):
            if NEXT == 0:
                NEXT = 1
            else:
                M = image[i - 1, j] + image[i, j] + image[i + 1, j] if 0 < i < h - 1 else 1
                if image[i, j] == 0 and M != 0:
                    a = [0] * 9
                    for k in range(3):
                        for l in range(3):
                            if -1 < (i - 1 + k) < h and -1 < (j - 1 + l) < w and image[i - 1 + k, j - 1 + l] == 255:
                                a[k * 3 + l] = 1
                    sum = a[0] * 1 + a[1] * 2 + a[2] * 4 + a[3] * 8 + a[5] * 16 + a[6] * 32 + a[7] * 64 + a[8] * 128
                    image[i, j] = array[sum] * 255
                    if array[sum] == 1:
                        NEXT = 0
    return image


def thinning(num=10):
     
     
    global img
    array = [0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, \
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, \
             0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, \
             1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, \
             1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0]

     
     
    for i in range(num):
        VThin(img, array)
        HThin(img, array)
     


def ridge_freq(im, mask, orient, blksze, windsze, minWaveLength, maxWaveLength):
    rows, cols = im.shape
    freq = np.zeros((rows, cols))

    for r in range(0, rows - blksze, blksze):
        for c in range(0, cols - blksze, blksze):
            blkim = im[r:r + blksze][:, c:c + blksze]
            blkor = orient[r:r + blksze][:, c:c + blksze]

            freq[r:r + blksze][:, c:c + blksze] = frequest(blkim, blkor, windsze, minWaveLength, maxWaveLength)

    freq = freq * mask
    freq_1d = np.reshape(freq, (1, rows * cols))
    ind = np.where(freq_1d > 0)

    ind = np.array(ind)
    ind = ind[1, :]

    non_zero_elems_in_freq = freq_1d[0][ind]

    meanfreq = np.mean(non_zero_elems_in_freq)
    medianfreq = np.median(non_zero_elems_in_freq)   
    return freq, meanfreq



def SeparateBackground(img):
    thresh = 0.1
    BLOCK_SIZE = 16
    rows,cols = img.shape
    rows = np.int32(BLOCK_SIZE * np.ceil((np.float32(rows)) / (np.float32(BLOCK_SIZE))))
    cols = np.int32(BLOCK_SIZE * np.ceil((np.float32(cols)) / (np.float32(BLOCK_SIZE))))
    padded_img = np.zeros((rows, cols))
    stddevim = np.zeros((rows, cols))

    padded_img[0:rows][:, 0:cols] = img

    for i in range(0, rows, BLOCK_SIZE):
        for j in range(0, cols, BLOCK_SIZE):
            block = padded_img[i:i + BLOCK_SIZE][:, j:j + BLOCK_SIZE]

            stddevim[i:i + BLOCK_SIZE][:, j:j + BLOCK_SIZE] = np.std(block) * np.ones(block.shape)

    stddevim = stddevim[0:rows][:, 0:cols]
    print(stddevim.mean())
    mask = stddevim < thresh
    padded_img[mask]=0
    return padded_img

def choosePic():
    global img
    img_path = askopenfilename(initialdir='./', title='选择待识别图片',
                               filetypes=[("tif", "*.tif"), ("jpg", "*.jpg"), ("png", "*.png")])
    if img_path:
        print(img_path)
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        print(type(img))
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转灰度图

        rows, cols = np.shape(img)
        aspect_ratio = np.double(rows) / np.double(cols)

        new_rows = 320  # randomly selected number
        new_cols = new_rows / aspect_ratio

        img = cv2.resize(img, (int(new_cols), int(new_rows)))
    root.destroy()



if __name__ == '__main__' :
    root = Tk()
    root.title("脊线特征处理")
    root.geometry('200x200')

    btnChoose = Button(root, text="选择图片", command=choosePic)
    btnChoose.place(x=70, y=70)
    root.mainloop()
    global img
    # img = cv2.imread('./DB3_B/106_4.tif')
    cv2.imshow('Raw Image',img)
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # img = cv2.resize(img,(320,320))
    image_enhance()
    print(img*1.0)
    print(np.unique(img))    
    # thinning()
    cv2.imshow('Ridge',img*1.0)
    feature()
    cv2.waitKey(0)