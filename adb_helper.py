#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : adb_helper
# @Date : 2022-07-28
# @Project: NC_Automation
# @AUTHOR : Totoro
from enum import IntEnum
from functools import wraps
import numpy as np

BLUESTACKS_ADB_ADDRESS = "127.0.0.1:5555"

QUICK_PAUSE = 0.8

RECOGNITION_ATTEMPTS = 8

OCR_TIME_OUT = 10

MIN_REFRESH_SCREEN_INTERVAL = 0.6


def Const(cls):
    @wraps(cls)
    def new_setattr(self, name, value):
        raise Exception('const : {} can not be changed'.format(name))

    cls.__setattr__ = new_setattr
    return cls


@Const
class Box(object):
    RELOAD_BOX = (1410, 854, 1748, 929)
    MAP_BOX = (208, 87, 522, 402)
    BIG_MAP_BOX = (910, 199, 1588, 879)


@Const
class Button(object):
    ANNOUNCE_CLOSE = (1891, 224)
    WORLD = (1029, 1028)
    WORLD_1 = (604, 527)
    BIG_MAP = (355, 198)


@Const
class LegendSubshape(object):
    CV_MASK = np.array([0, 0, 0, 1, 2, 0], dtype=np.int32)
    BB_MASK = np.array([0, 0, 0, 1, 1, 0], dtype=np.int32)
    CC_MASK = np.array([0, 0, 0, 1, 0, 1], dtype=np.int32)
    DD_MASK = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)


@Const
class Filters(object):
    WHITE_LOW = np.array([0, 0, 0], dtype=np.uint8)
    WHILE_HIGH = np.array([0, 0, 255], dtype=np.uint8)
    GREEN_LOW = np.array([60, 180, 189], dtype=np.uint8)
    GREEN_HIGH = np.array([80, 200, 230], dtype=np.uint8)
    LINE_LOW = np.array([85, 40, 190], dtype=np.uint8)
    LINE_HIGH = np.array([100, 60, 210], dtype=np.uint8)


@Const
class GameStatus(object):
    PREPARE_0 = 0
    PREPARE_1 = 1
    STARTED = 2
    LEGEND = 3
    RESULT = 4


@Const
class SteerStatus(object):
    NONE = 0
    LEFT_FULL = 1
    LEFT_HALF = 2
    FORWARD = 3
    RIGHT_HALF = 4
    RIGHT_FULL = 5


@Const
class PlayerSide(object):
    SELF = 0
    ALLY = 1
    ENEMY = 2


@Const
class ShipType(object):
    CV = 0
    BB = 1
    CC = 2
    DD = 3


@Const
class _Const(object):
    Box = Box()
    Button = Button()
    Filters = Filters()
    GameStatus = GameStatus()
    SteerStatus = SteerStatus()
    PlayerSide = PlayerSide()
    ShipType = ShipType()


CONST = _Const()
