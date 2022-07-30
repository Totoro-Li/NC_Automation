#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : utility
# @Date : 2022-07-28
# @Project: NC_Automation
# @AUTHOR : Totoro


import cv2
import os
import subprocess
import numpy as np


def get_poly_center(vertexes: list) -> tuple:
    _x_list = [vertex[0][0] for vertex in vertexes]
    _y_list = [vertex[0][1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return _x, _y


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
    return x, y


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


def get_mode(in_list: list):
    counts = np.bincount(in_list)
    return np.argmax(counts)
