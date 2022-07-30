#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : FLANN_test
# @Date : 2022-07-29
# @Project: NC_Automation
# @AUTHOR : Totoro

import cv2
from utility import FLANN_feature_match

template = cv2.imread("NC_AppIcon.png", cv2.IMREAD_COLOR)  # queryImage
target = cv2.imread("asfg.png", cv2.IMREAD_COLOR)  # trainImage
cv2.imwrite("dhyr.png",target)
dst = FLANN_feature_match(queryImage=template, trainingImage=target, min_match_count=10)
print(dst[0][0])