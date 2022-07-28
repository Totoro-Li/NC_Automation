#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : adb_helper
# @Date : 2022-07-28
# @Project: NC_Automation
# @AUTHOR : Totoro
from enum import IntEnum

BLUESTACKS_ADB_ADDRESS = "127.0.0.1:5555"

ANNOUNCE_CLOSE = (1891, 224)
WORLD = (1029, 1028)
WORLD_1 = (604, 527)

QUICK_PAUSE = 1

RECOGNITION_ATTEMPTS = 8

RELOAD_BOX = (1410, 854, 1748, 929)

OCR_TIME_OUT = 10


class GameStatus(IntEnum):
    PREPARE_0 = 0
    PREPARE_1 = 1
    STARTED = 2
    LEGEND = 3
    RESULT = 4
