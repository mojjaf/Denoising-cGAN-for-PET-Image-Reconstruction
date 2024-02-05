import numpy as np
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import uniform_filter
from skimage._shared.utils import _supported_float_type, check_shape_equality, warn
from skimage.util.arraycrop import crop
from skimage.util.dtype import dtype_range
from skimage.metrics import normalized_root_mse
from itertools import product
import time




def getNRMSE(A,B):
    A=np.array(A)
    B=np.array(B)
    return normalized_root_mse(B, A, normalization='mean')

def getPSNR(A, B):
    # A is input, B is target
    mse = mean_squared_error(A, B)
    return 20*np.log10(np.max(B)) - 10*np.log10(mse)


def getSSIM(A, B, randomWindows=False, windowRatio=4, randomSeed=None):
    # A is input, B is target
    dataRange = np.max(B)-np.min(B)
    if randomWindows:
        return ssimRandomWindows(A, B, data_range=dataRange, randomSeed=randomSeed, windowRatio=windowRatio)
    else:
        return ssim(A, B, data_range=dataRange)


def getPixels(image, idx):
    # Return a list of pixel values
    # image: a 2-D list or a 2-D numpy array
    # idx: list of tuples (x, y) or a 2-D numpy array of dimension (numIndices, 2)
    pixels = []
    if type(idx) == list:
        for i in idx:
            pixels.append(image[i[0], i[1]])
    elif type(idx) == np.ndarray:
        for i in range(idx.shape[0]):
            pixels.append(image[idx[i, 0], idx[i, 1]])
    else:
        print("index type not supported")
        assert (False)
    return pixels


def getContrast(data, roi, lesionBox, roiBox, bgBox):
    # Input
    #   data: 2-D numpy array
    #   roi: tuple (x, y), center of lesion, roi, and bg
    #   lesionBox: tuple (width, height)
    #   roiBox: tuple (width, height)
    #   bgBox: tuple (width, height)
    # Note
    #  Assume lesion center, roi center, bg center are all the same.
    # Output
    #   contrast: float
    assert (roi[0] > 0 and roi[1] > 0)
    assert (lesionBox[0] > 0 and lesionBox[1] > 0)
    assert (roiBox[0] > 0 and roiBox[1] > 0)
    assert (bgBox[0] > 0 and bgBox[1] > 0)
    l_x1, l_x2, l_y1, l_y2 = (int(roi[0]-roiBox[0]/2.0), int(
        roi[0]+lesionBox[0]/2.0), int(roi[1]-lesionBox[1]/2.0), int(roi[1]+lesionBox[1]/2.0))
    r_x1, r_x2, r_y1, r_y2 = (int(roi[0]-roiBox[0]/2.0), int(
        roi[0]+roiBox[0]/2.0), int(roi[1]-roiBox[1]/2.0), int(roi[1]+roiBox[1]/2.0))
    b_x1, b_x2, b_y1, b_y2 = (int(roi[0]-bgBox[0]/2.0), int(
        roi[0]+bgBox[0]/2.0), int(roi[1]-bgBox[1]/2.0), int(roi[1]+bgBox[1]/2.0))
    signal = np.mean(data[l_x1:l_x2, l_y1:l_y2])
    background = (np.sum(data[b_x1:b_x2, b_y1:b_y2]) - np.sum(data[r_x1:r_x2, r_y1:r_y2])) / \
        (bgBox[0]*bgBox[1]-roiBox[0]*roiBox[1])
    contrast = (signal-background)/background
    return contrast


def getContrastPixels(roiPixels, bgPixels):
    # Get Contrast from roi pixel values and background pixel values.
    signal = np.mean(roiPixels)
    background = np.mean(bgPixels)
    contrast = (signal-background)/background
    return contrast


def getNCRC(A, B, roi, lesionBox, roiBox, bgBox):
    # A input B target
    contrastA = getContrast(A, roi, lesionBox, roiBox, bgBox)
    contrastB = getContrast(B, roi, lesionBox, roiBox, bgBox)
    return contrastA/contrastB


def getNCRCPixels(inputRoiPixels, inputBgPixels, targetRoiPixels, targetBgPixels):
    # A input B target
    contrastInput = getContrastPixels(inputRoiPixels, inputBgPixels)
    contrastTarget = getContrastPixels(targetRoiPixels, targetBgPixels)
    return contrastInput/contrastTarget


def getCNR(data, roi, lesionBox, roiBox, bgBox):
    l_x1, l_x2, l_y1, l_y2 = (int(roi[0]-lesionBox[0]/2.0), int(
        roi[0]+lesionBox[0]/2.0), int(roi[1]-lesionBox[1]/2.0), int(roi[1]+lesionBox[1]/2.0))
    r_x1, r_x2, r_y1, r_y2 = (int(roi[0]-roiBox[0]/2.0), int(
        roi[0]+roiBox[0]/2.0), int(roi[1]-roiBox[1]/2.0), int(roi[1]+roiBox[1]/2.0))
    b_x1, b_x2, b_y1, b_y2 = (int(roi[0]-bgBox[0]/2.0), int(
        roi[0]+bgBox[0]/2.0), int(roi[1]-bgBox[1]/2.0), int(roi[1]+bgBox[1]/2.0))
    signal = np.mean(data[l_x1:l_x2, l_y1:l_y2])
    background = (np.sum(data[b_x1:b_x2, b_y1:b_y2]) - np.sum(data[r_x1:r_x2, r_y1:r_y2])) / \
        (bgBox[0]*bgBox[1]-roiBox[0]*roiBox[1])
    contrast = (signal-background)/background
    noise = (((np.sum((data[b_x1:b_x2, b_y1:b_y2] - background)**2) - np.sum(
        (data[r_x1:r_x2, r_y1:r_y2] - background)**2)) / (bgBox[0]*bgBox[1]-roiBox[0]*roiBox[1]))**0.5)/background
    cnr = contrast/noise
    return cnr


def getCNRPixels(inputRoiPixels, inputBgPixels):
    signal = np.mean(inputRoiPixels)
    background = np.mean(inputBgPixels)
    noise = np.std(inputBgPixels)/background
    contrast = (signal-background)/background
    cnr = contrast/noise
    return cnr


def ssimRandomWindows(im1, im2,
                      *,
                      windowRatio=4,
                      randomSeed=0,
                      win_size=None, gradient=False, data_range=None,
                      channel_axis=None, multichannel=False,
                      gaussian_weights=False, full=False, **kwargs):
    # Returns SSIM calculated by averaging structural similarity of randomly selected windows
    # windowRatio: random number of windows to total number of windows. If windowRatio=2, then half of the windows are randomly selected and used for SSIM calculation.
    check_shape_equality(im1, im2)
    float_type = _supported_float_type(im1.dtype)

    K1 = kwargs.pop('K1', 0.01)
    K2 = kwargs.pop('K2', 0.03)
    sigma = kwargs.pop('sigma', 1.5)
    if K1 < 0:
        raise ValueError("K1 must be positive")
    if K2 < 0:
        raise ValueError("K2 must be positive")
    if sigma < 0:
        raise ValueError("sigma must be positive")
    use_sample_covariance = kwargs.pop('use_sample_covariance', True)

    if win_size is None:
        win_size = 7   # backwards compatibility

    if np.any((np.asarray(im1.shape) - win_size) < 0):
        raise ValueError(
            'win_size exceeds image extent. '
            'Either ensure that your images are '
            'at least 7x7; or pass win_size explicitly '
            'in the function call, with an odd value '
            'less than or equal to the smaller side of your '
            'images. If your images are multichannel '
            '(with color channels), set channel_axis to '
            'the axis number corresponding to the channels.')

    if not (win_size % 2 == 1):
        raise ValueError('Window size must be odd.')

    if data_range is None:
        if im1.dtype != im2.dtype:
            warn("Inputs have mismatched dtype.  Setting data_range based on "
                 "im1.dtype.", stacklevel=2)
        dmin, dmax = dtype_range[im1.dtype.type]
        data_range = dmax - dmin

    ndim = im1.ndim

    filter_func = uniform_filter
    filter_args = {'size': win_size}

    # ndimage filters need floating point data
    im1 = im1.astype(float_type, copy=False)
    im2 = im2.astype(float_type, copy=False)

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute (weighted) means
    ux = filter_func(im1, **filter_args)
    uy = filter_func(im2, **filter_args)

    # compute (weighted) variances and covariances
    uxx = filter_func(im1 * im1, **filter_args)
    uyy = filter_func(im2 * im2, **filter_args)
    uxy = filter_func(im1 * im2, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    # to avoid edge effects will ignore filter radius strip around edges
    pad = (win_size - 1) // 2

    # compute (weighted) mean of ssim. Use float64 for accuracy.
    data = crop(S, pad).flatten()
    l = data.shape[0]
    np.random.seed(randomSeed)
    mssim = np.random.choice(
        data, l//windowRatio, replace=True).mean(dtype=np.float64)

    if full:
        return mssim, S
    else:
        return mssim


def generateGrid(numResamples, imageSize, samplingRatio, holeSize=None, baseRandomSeed=0, debug=False):
    # holeSize: if not None, then the grid will be generated with a hole of size holeSize at the center of the image
    if debug:
        print("Precalculating random grids per sample...")
        gridTime = time.time()
    if holeSize is None:
        imageGrid_np = np.zeros(
            (numResamples, imageSize[0]*imageSize[1]//samplingRatio, 2), dtype=np.int32)  # (numResamples, numPixels, 2)
        for s in range(numResamples):
            randomSeed = baseRandomSeed + s
            np.random.seed(randomSeed)
            choice = np.random.choice(
                imageSize[0]*imageSize[1], imageSize[0]*imageSize[1]//samplingRatio, replace=True)
            imageGrid_np[s] = np.array(
                list(product(range(imageSize[0]), range(imageSize[1]))))[choice]
    else:
        assert (holeSize[0] < imageSize[0])
        assert (holeSize[1] < imageSize[1])
        assert (len(holeSize) == 2)
        imageGrid_np = np.zeros(
            (numResamples, (imageSize[0]*imageSize[1]-holeSize[0]*holeSize[1])//samplingRatio, 2), dtype=np.int32)  # (numResamples, numPixels, 2)
        for s in range(numResamples):
            randomSeed = baseRandomSeed + s
            np.random.seed(randomSeed)
            choice = np.random.choice(
                imageSize[0]*imageSize[1]-holeSize[0]*holeSize[1], (imageSize[0]*imageSize[1]-holeSize[0]*holeSize[1])//samplingRatio, replace=True)
            backgroundGrid = np.array(
                list(product(range(imageSize[0]), range(imageSize[1])))).reshape((-1, 2))
            # width of the left/right strip
            width = (imageSize[0] - holeSize[0])//2
            grid = backgroundGrid[np.logical_or.reduce([backgroundGrid[:, 0] < width, backgroundGrid[:, 0] >=
                                  imageSize[0]-width, backgroundGrid[:, 1] < width, backgroundGrid[:, 1] >= imageSize[1]-width])]
            imageGrid_np[s] = grid[choice, :]
    if debug:
        print("Precalculating random grids took {} seconds.".format(
            time.time()-gridTime))
    return imageGrid_np

# def bootstrapAssert(inputs_list, targets_list):


def bootstrapAssert(inputs_list, targets_list, rois_np, lesionBox, roiBox, bgBox):
    assert (len(inputs_list) > 0)
    assert (len(inputs_list) == len(targets_list))
    for i in range(len(inputs_list)):
        assert (inputs_list[i].shape == targets_list[i].shape)
        if i > 0:
            assert (inputs_list[i-1].shape == inputs_list[i].shape)
    if rois_np is not None:
        assert (len(inputs_list) == rois_np.shape[0])
        assert (rois_np.shape[1] == len(lesionBox)
                == len(roiBox) == len(bgBox) == 2)
        assert (lesionBox[0] > 0 and lesionBox[1] > 0)
        assert (roiBox[0] > 0 and roiBox[1] > 0)
        assert (bgBox[0] > 0 and bgBox[1] > 0)


def bootstrap(inputs_list, targets_list, func, numResamples, samplingRatio=4, metric=None, baseRandomSeed=0, debug=False):
    # Description:
    #     A higher order function that takes a function to calculate mean and standard error by bootstrapping. This function calculates the mean of func and a standard error calculated based on bootstrapping method. Note that the standard error includes only the measurement error, not the sampling error.
    # https://thestatsgeek.com/2013/07/02/the-miracle-of-the-bootstrap/
    # Inputs:
    #     inputs_list: A list of 2-D numpy arrays (images)
    #     targets_list: A list of 2-D numpy arrays (images)
    #     func: function to be used for calculating metric
    #     metric: A string that specifies the metric to be used. If None, func will be used. Currently supports "SSIM", "NCRC", and "CNR".
    #     numResamples: Number of resampling for bootstrapping.
    #     samplingRatio: Ratio of number of pixels (windows in case of SSIM) to be used for bootstrapping. For example, if samplingRatio is 2, then 1/2 of the pixels per image will be used for bootstrapping.
    #     baseRandomSeed: Base random seed for numpy. For each resampling, random seed is set to baseRandomSeed + resampling index.
    # Outputs:
    #     The mean of the metric, the standard error estimated by bootstrapping.

    # Assert data format
    bootstrapAssert(inputs_list, targets_list, None, None, None, None)
    assert (metric in [None, "SSIM"])

    numImages = len(inputs_list)
    imageSize = inputs_list[0].shape  # Should be a tuple of two integers
    # numPixels = imageSize[0]*imageSize[1]

    # 1. Calculate the estimator - the mean of the metric
    metrics_list = []
    for i in range(numImages):
        metrics_list.append(func(inputs_list[i], targets_list[i]))
    m = np.mean(metrics_list)

    # 2. Calculate the standard error of the estimator
    # First, precalculate random grids per sample
    if metric != "SSIM":
        imageGrid_np = generateGrid(
            numResamples, imageSize, samplingRatio, baseRandomSeed=0, debug=False)
    else:
        # don't need to precalculate random grids
        pass

    # Second, calculate the standard error
    if debug:
        print("Calculating the standard error...")
        stdTime = time.time()
    stdPerImage_list = []  # The mean of std per patient
    for i in range(numImages):
        metricsPerImagePerSample_list = []
        for s in range(numResamples):
            startTime = time.time()
            randomSeed = baseRandomSeed + s
            if metric == "SSIM":
                metricsPerImagePerSample_list.append(func(
                    inputs_list[i], targets_list[i], randomWindows=True, windowRatio=samplingRatio, randomSeed=randomSeed))

            else:
                inputPixels = getPixels(inputs_list[i], imageGrid_np[s])
                targetPixels = getPixels(targets_list[i], imageGrid_np[s])
                metricsPerImagePerSample_list.append(func(
                    inputPixels, targetPixels))
            if debug:
                print("Resampling {}/{} for image {}/{} took {} seconds.".format(s,
                      numResamples, i, numImages, time.time()-startTime))
        stdPerImage = np.std(metricsPerImagePerSample_list) / \
            np.sqrt(samplingRatio)
        stdPerImage_list.append(stdPerImage)
    se = np.mean(stdPerImage_list)
    if debug:
        print("Calculating the standard error took {} seconds.".format(
            time.time()-stdTime))
    return m, se


def bootstrapNCRC(inputs_list, targets_list, rois_np, lesionBox, roiBox, bgBox, numResamples=10, samplingRatio=4, baseRandomSeed=0, debug=False):
    # Inputs:
    #     rois_np: A numpy array (images) of the centers of ROIs (numImages, 2)
    #     lesionBox: A tuple of two integers (width, height) of width of the lesion box. should be smaller than the size of the lesion.
    #     roiBox: A tuple of two integers (width, height) of width of the ROI box. should be bigger than the lesion box but not too big. This box will be used for PSNR, SSIM, and background definition. For background, this number will be the inner box of the background box.
    #     bgBox: A tuple of two integers (width, height) of width of the background box. should be bigger than the ROI box. This box will be used for outer box of the background.
    # Assert data format
    bootstrapAssert(inputs_list, targets_list,
                    rois_np, lesionBox, roiBox, bgBox)

    numImages = len(inputs_list)

    # 1. Calculate the estimator - the mean of the metric
    metrics_list = []
    for i in range(numImages):
        metrics_list.append(getNCRC(
            inputs_list[i], targets_list[i], rois_np[i], lesionBox, roiBox, bgBox))
    m = np.mean(metrics_list)

    # 2. Calculate the standard error of the estimator
    # need two random grids, one for roi and one for background
    imageGridRoi_np = generateGrid(
        numResamples, lesionBox, samplingRatio, baseRandomSeed=baseRandomSeed, debug=False)
    imageGridBg_np = generateGrid(
        numResamples, bgBox, samplingRatio, holeSize=roiBox, baseRandomSeed=baseRandomSeed, debug=False)

    # Second, calculate the standard error
    if debug:
        print("Calculating the standard error...")
        stdTime = time.time()
    stdPerImage_list = []  # The mean of std per patient
    for i in range(numImages):
        metricsPerImagePerSample_list = []
        for s in range(numResamples):
            perResampleTime = time.time()
            # need to create a list of 2-D coordinates out of imageGridRoi_np and imageGridBg_np.
            roiPixels_np = imageGridRoi_np[s, :, :] + \
                rois_np[i, :] - roiBox[0]//2  # (numPixels, 2)
            bgPixels_np = imageGridBg_np[s, :, :] + \
                rois_np[i, :] - bgBox[0]//2  # (numPixels, 2)
            inputRoiPixels = getPixels(inputs_list[i], roiPixels_np)
            inputBgPixels = getPixels(inputs_list[i], bgPixels_np)
            targetRoiPixels = getPixels(targets_list[i], roiPixels_np)
            targetBgPixels = getPixels(targets_list[i], bgPixels_np)
            metricsPerImagePerSample_list.append(getNCRCPixels(
                inputRoiPixels, inputBgPixels, targetRoiPixels, targetBgPixels))
            if debug:
                print("Resampling {}/{} for image {}/{} took {} seconds.".format(s,
                      numResamples, i, numImages, time.time()-perResampleTime))
        stdPerImage = np.std(metricsPerImagePerSample_list) / \
            np.sqrt(samplingRatio)
        stdPerImage_list.append(stdPerImage)
    se = np.mean(stdPerImage_list)
    if debug:
        print("Calculating the standard error took {} seconds.".format(
            time.time()-stdTime))
    return m, se


def bootstrapCNR(inputs_list, rois_np, lesionBox, roiBox, bgBox, numResamples=10, samplingRatio=4, baseRandomSeed=0, debug=False):
    # Inputs:
    #     rois_np: A numpy array (images) of the centers of ROIs (numImages, 2)
    #     bgs_np: A numpy array (images) of the centers of backgrounds (numImages, 2)
    # Assert data format
    bootstrapAssert(inputs_list, inputs_list,
                    rois_np, lesionBox, roiBox, bgBox)

    numImages = len(inputs_list)

    # 1. Calculate the estimator - the mean of the metric
    metrics_list = []
    for i in range(numImages):
        cnr = getCNR(inputs_list[i], rois_np[i], lesionBox, roiBox, bgBox)
        metrics_list.append(cnr)
    m = np.mean(metrics_list)

    # 2. Calculate the standard error of the estimator
    # need two random grids, one for roi and one for background
    imageGridRoi_np = generateGrid(
        numResamples, lesionBox, samplingRatio, baseRandomSeed=baseRandomSeed, debug=False)
    imageGridBg_np = generateGrid(
        numResamples, bgBox, samplingRatio, holeSize=roiBox, baseRandomSeed=baseRandomSeed, debug=False)

    # Second, calculate the standard error
    if debug:
        print("Calculating the standard error...")
        stdTime = time.time()
    stdPerImage_list = []  # The mean of std per patient
    for i in range(numImages):
        metricsPerImagePerSample_list = []
        for s in range(numResamples):
            perResampleTime = time.time()
            # need to create a list of 2-D coordinates out of imageGridRoi_np and imageGridBg_np.
            roiPixels_np = imageGridRoi_np[s, :, :] + \
                rois_np[i, :] - roiBox[0]//2  # (numPixels, 2)
            bgPixels_np = imageGridBg_np[s, :, :] + \
                rois_np[i, :] - bgBox[0]//2  # (numPixels, 2)
            inputRoiPixels = getPixels(inputs_list[i], roiPixels_np)
            inputBgPixels = getPixels(inputs_list[i], bgPixels_np)
            metricsPerImagePerSample_list.append(getCNRPixels(
                inputRoiPixels, inputBgPixels))
            if debug:
                print("Resampling {}/{} for image {}/{} took {} seconds.".format(s,
                      numResamples, i, numImages, time.time()-perResampleTime))
        stdPerImage = np.std(metricsPerImagePerSample_list) / \
            np.sqrt(samplingRatio)
        stdPerImage_list.append(stdPerImage)
    se = np.mean(stdPerImage_list)
    if debug:
        print("Calculating the standard error took {} seconds.".format(
            time.time()-stdTime))
    return m, se
