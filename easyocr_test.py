#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : easyocr_test
# @Date : 2022-07-29
# @Project: NC_Automation
# @AUTHOR : Totoro
import cv2
from PIL import Image
from utility import cut_image_internal, extract_text_internal

if __name__ == "__main__":
    image = cv2.imread(u"./test.png", flags=cv2.IMREAD_COLOR)
    cropped = cut_image_internal(image=image, box=(0, 0, 100, 100))
    # cv2.imwrite("cropped.png", cropped)
    print(extract_text_internal(image=cropped, num=False))
