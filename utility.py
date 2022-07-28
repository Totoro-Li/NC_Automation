#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : utility
# @Date : 2022-07-28
# @Project: NC_Automation
# @AUTHOR : Totoro
from PIL import Image
import cv2
import easyocr
import numpy as np

# OCR Recognition
def cut_image_internal(filename=None, outname="out.png", box=None):
    # (left, upper, right, lower) means
    # (left, upper)---------------------->x
    # |
    # |
    # |
    # |
    # v----------------------------(right, lower)
    # y
    if filename is None:
        raise ValueError("Empty filename.")
    image = Image.open(filename)
    if box is None:
        box = (0, 0, image.size[0], image.size[1])

    cropped = image.crop(box)  # 设置图像裁剪区域(left, upper, right, lower)
    cropped.save(outname)


def extract_text_internal(filename=None, num=False):
    # 图像分割
    allow_list = None
    if filename is None:
        raise ValueError("Empty filename.")
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转灰度图
    (T, threshInv) = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY_INV)  # 反阈值化，阈值为215
    cv2.imwrite(filename, threshInv)
    reader = easyocr.Reader(['en'], gpu=True)  # 识别中文和英文，不用GPU
    if num:
        allow_list = '0123456789'
    result = reader.readtext(filename, allowlist=allow_list)
    return result


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

