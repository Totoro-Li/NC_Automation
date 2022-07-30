#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : visual
# @Date : 2022-07-30
# @Project: NC_Automation
# @AUTHOR : Totoro
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import easyocr
from utility import tap_internal, get_poly_center


# OCR Recognition
def cut_image_internal(image=None, box=None):
    """
    In OpenCV, index retrieving of image requires y coordinate before x coordinate, same
    as matrix index.
    :param image:
    :param box:
    :return: image
    """
    # (left, upper, right, lower) means
    # (left, upper)---------------------->x
    # |
    # |
    # |
    # |
    # v----------------------------(right, lower)
    # y
    if image is None:
        raise ValueError("Empty image file")
    h, w = image.shape[:2]
    if box is None:
        box = (0, 0, w, h)
    cropped = image[box[1]:box[3], box[0]:box[2]]  # set crop box [start_row:end_row, start_col:end_col]
    return cropped


def extract_text_internal(image=None, num=False):
    allow_list = None
    if image is None:
        raise ValueError("Empty image file for extraction")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (T, threshInv) = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("./picture/cropped/textarea.png", threshInv)
    reader = easyocr.Reader(['en'], gpu=True)
    if num:
        allow_list = '0123456789'
    result = reader.readtext("./picture/cropped/textarea.png", allowlist=allow_list)
    return result


def locate_pic_internal(query_image=None, training_image=None, desc=None,
                        click=False, device=None, min_match_count=10, crop_box=None):
    dst = None
    if crop_box is not None:
        query_image = cut_image_internal(image=query_image, box=crop_box)
    dst = FLANN_feature_match(queryImage=query_image, trainingImage=training_image, min_match_count=min_match_count)
    if dst is not None:
        if desc is not None:
            print(desc)
        if click:
            tap_internal(get_poly_center(dst), device=device)
            return dst
    return dst


def line_detect_internal(image=None, filter_low=None, filter_high=None):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, filter_low, filter_high)
    cv2.imshow('sdf', mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 15  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(image) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    # Output "lines" is an array containing endpoints of detected line segments
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    return lines


# Stream
def bytes_to_ndarray(bytes):
    bytes_io = bytearray(bytes)
    img = Image.open(BytesIO(bytes_io))
    return np.array(img)


def PIL2OpenCV(pil_img):
    numpy_image = np.array(pil_img)

    # convert to a openCV2 image, notice the COLOR_RGB2BGR which means that
    # the color is converted from RGB to BGR format
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return opencv_image


def OpenCV2PIL(opencv_image):
    # convert from openCV2 to PIL. Notice the COLOR_BGR2RGB which means that
    # the color is converted from BGR to RGB
    color_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(color_coverted)
    return pil_image


def legend_match_internal(img, low, high) -> (np.array(), int, int):
    """
    :param img: input cv2 image object in BGR
    :param low: low threshold for color filtering
    :param high: high threshold for color filtering
    :return: (subshape_sum, center_x, center_y)
    """
    if img is None:
        raise ValueError("Empty image input!")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert from BGR to HSV color space
    # Range of each value in OpenCV HSV color space
    # H:0-180
    # S:0-255
    # V:0-255
    mask = cv2.inRange(hsv, low, high)
    cv_kernel = np.ones((5, 5), dtype=np.uint8)
    mask_morphed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=cv_kernel, iterations=2)
    cnts = cv2.findContours(mask_morphed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None  # Figure to cover centroid
    for cnt in cnts:
        ((center_x, center_y), radius) = cv2.minEnclosingCircle(cnt)
        # Calculate moment
        # M = cv2.moments(cnt)
        # Calculate centroid
        cv2.circle(img, (int(center_x), int(center_y)), int(radius), (0, 255, 255), 2)
        cv2.circle(img, center, 5, (0, 0, 255), -1)
        x, y, w, h = cv2.boundingRect(cnt)
        cnt_img = cut_image_internal(image=mask, box=(x, y, x + w, y + h))
        elements = cv2.findContours(cnt_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        subshape_sum = np.zeros(6, dtype=np.int32)
        for element in elements:
            approx = cv2.approxPolyDP(element, 0.04 * cv2.arcLength(element, True), True)
            if len(approx) <= 5:
                subshape_sum[len(approx)] += 1
        return subshape_sum, int(center_x), int(center_y)


def FLANN_feature_match(queryImage, trainingImage, min_match_count=10):
    """

    :param min_match_count: Minimum feature matching count
    :param queryImage: Small image containing expected feature
    :param trainingImage: Big image for finding features in
    """
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(queryImage, None)
    kp2, des2 = sift.detectAndCompute(trainingImage, None)
    # Set FLANN matching
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    # Drop matches distance greater than 0.7
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) > min_match_count:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        transformation_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()
        h, w = queryImage.shape[:2]

        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, transformation_matrix)
        return dst
        # cv2.polylines(trainingImage, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
    else:
        print("Not enough matches are found - %d/%d" % (len(good), min_match_count))
        matches_mask = None
        return None
