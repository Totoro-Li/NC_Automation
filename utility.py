#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : utility
# @Date : 2022-07-28
# @Project: NC_Automation
# @AUTHOR : Totoro
from io import BytesIO

from PIL import Image
import cv2
import os
import easyocr
import subprocess
import numpy as np


# OCR Recognition
def cut_image_internal(image=None, box=None):
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
    h, w, _ = image.shape
    if box is None:
        box = (0, 0, w, h)
    cropped = image[box[1]:box[3], box[0]:box[2]]  # set crop box [start_row:end_row, start_col:end_col]
    return cropped


def extract_text_internal(image=None, num=False):
    allow_list = None
    if image is None:
        raise ValueError("Empty image file for extraction")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度图
    (T, threshInv) = cv2.threshold(image, 160, 255, cv2.THRESH_BINARY_INV)  # 反阈值化，阈值为215
    cv2.imwrite("./picture/cropped/textarea.png", threshInv)
    reader = easyocr.Reader(['en'], gpu=True)  # 识别中文和英文，不用GPU
    if num:
        allow_list = '0123456789'
    result = reader.readtext("./picture/cropped/textarea.png", allowlist=allow_list)
    return result


def get_poly_center(vertexes: list) -> tuple:
    _x_list = [vertex[0][0] for vertex in vertexes]
    _y_list = [vertex[0][1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return _x, _y


def locate_pic_internal(query_image=None, training_image=None, desc=None, click=False, device=None, min_match_count=10):
    """
    loop not suggested to use since screenshot is not updated each time
    :rtype: object
    """
    dst = None
    dst = FLANN_feature_match(queryImage=query_image, trainingImage=training_image, min_match_count=min_match_count)
    if dst is not None:
        if desc is not None:
            print(desc)
        if click:
            tap_internal(get_poly_center(dst), device=device)
            return dst
    return dst


# ADB
def adb_send_internal(device=None, command=None):
    if command is None:
        raise ValueError("Empty command")
    if device is None:
        os.system(f"adb {command}")
    else:
        os.system(f"adb -s {device} {command}")


def tap_internal(x: int or tuple, y=None, device=None):
    if y is None:
        x, y = x
    adb_send_internal(device=device, command=f"shell input tap {x} {y}")


# Screenshot
def screenshot_internal(device=None):
    pipe = subprocess.Popen(f"adb -s {device} shell screencap -p",
                            stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, shell=True)
    image_bytes = pipe.stdout.read().replace(b'\r\n', b'\n')
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    return image


def screenshot_save(device=None, filename="default.png", path="/sdcard"):
    adb_send_internal(device=device, command=f"shell mkdir -p {path}")
    adb_send_internal(device=device, command=f"shell screencap -p {path}/{filename}")
    adb_send_internal(device=device, command=f"pull {path}/{filename}")


def get_bin_indice(value, bins):
    return np.digitize(value, bins)


def get_bin_target(value, bins, targets):
    if len(targets) == len(bins):
        return list(targets[i] for i in np.digitize(value, bins))
    else:
        raise ValueError('Targets do not match all n+1 bin intervals to output!')


def in_value_tolerance(value: list, mid: int, error: int) -> list:
    return list(get_bin_indice(x, [mid - error, mid, mid + error]) in [1, 2] for x in value)


def isNan(value) -> bool:
    return value != value


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
        # 获取关键点的坐标
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # 计算变换矩阵和MASK
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = queryImage.shape
        # 使用得到的变换矩阵对原图像的四个角进行变换，获得在目标图像上对应的坐标
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        return dst
        # cv2.polylines(trainingImage, [np.int32(dst)], True, 0, 2, cv2.LINE_AA)
    else:
        # print("Not enough matches are found - %d/%d" % (len(good), min_match_count))
        matchesMask = None
        return None
