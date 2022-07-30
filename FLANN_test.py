#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : FLANN_test
# @Date : 2022-07-29
# @Project: NC_Automation
# @AUTHOR : Totoro

import cv2
from visual import FLANN_feature_match

template = cv2.imread("./picture/2500x1080/NC_AppIcon.png", cv2.IMREAD_COLOR)  # queryImage
target = cv2.imread("./picture/samples/FLANN_sample.png", cv2.IMREAD_COLOR)  # trainImage
cv2.imwrite("FLANN_sample_out.png", target)
dst = FLANN_feature_match(queryImage=template, trainingImage=target, min_match_count=10)
print(dst[0][0])
