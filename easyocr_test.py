#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : easyocr_test
# @Date : 2022-07-29
# @Project: NC_Automation
# @AUTHOR : Totoro
from PIL import Image
from utility import cut_image_internal, extract_text_internal

if __name__ == "__main__":
    cut_image_internal(filename=u"./picture/test.png", outname=u"test_out.png")
    print(extract_text_internal(filename=u"test_out.png", num=False))
