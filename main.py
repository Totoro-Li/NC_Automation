import os
import sys

import numpy as np
import pyautogui as auto
import time
import configparser
from PIL import Image
from adb_helper import *
import utility
import asyncio
import cv2

'''属性设置'''
config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')
Apath = config.get('main', 'Apath')
filename = config.get('main', 'Filename')
pausetime = config.getint('main', 'PauseTime')
ip = config.get('main', 'IP')
c_confidence = config.getfloat('main', 'Confidence')
SC = config.getboolean('main', 'StoneCrush')


def adb_send(command):
    utility.adb_send_internal(ip, command)


def screenshot():
    return utility.screenshot_internal(ip)


def get_filename_by_id(id=0):
    return f"{id}_{filename}"


def getResolution():
    try:
        global SR  # ScreenResolution
        global RES
        img = screenshot()
        height, width, _ = img.shape
        SR = f'{width}x{height}'
        RES = (width, height)
        return SR
    except ValueError as e:
        print('ADB Connection error')
        sys.exit(0)


def center_tap():
    tap(RES[0] // 2, RES[1] // 2)


def cut_image(image, box):
    return utility.cut_image_internal(image, box=box)


def extract_text(image, num=False):
    return utility.extract_text_internal(image=image, num=num)


def tap(x: int or tuple, y=None):
    utility.tap_internal(x, y, ip)
    print(f"Tap event at point {x} {y}")


def locate_pic(query_image=None, training_image=None, desc=None, click=False, device=None, min_match_count=10):
    return utility.locate_pic_internal(query_image=query_image, training_image=training_image, click=click, device=ip, min_match_count=min_match_count)


def screen_change(pause=QUICK_PAUSE):
    im1 = screenshot()
    time.sleep(pause)
    im2 = screenshot()
    if im1.shape != im2.shape:
        raise ValueError("Change of device resolution or orientation.")
    difference = cv2.subtract(im1, im2)
    result = np.any(difference)
    return result


def locate_or_exit(template: str, click=False):
    p = None
    query_image = cv2.imread(f"./picture/{SR}/{template}", 0)  # trainImage
    for attempt in range(RECOGNITION_ATTEMPTS):
        p = locate_pic(query_image=query_image, training_image=screenshot(), click=click)
        if p is not None:
            print(p)
            return p
        time.sleep(pausetime)
    if p is None:
        raise ValueError("Recognition failure")
    return p


def wait_till_screen_static(timeout=5):
    start = time.time()
    while time.time() - start < timeout:
        if not screen_change(pausetime):
            break
        time.sleep(pausetime)


def start_game():
    locate_or_exit("NC_APPIcon.png", click=True)


def start_1_1():
    tap(WORLD)
    time.sleep(QUICK_PAUSE)
    tap(WORLD_1)
    time.sleep(QUICK_PAUSE)
    locate_or_exit("1-1.png", click=True)
    time.sleep(QUICK_PAUSE)
    locate_or_exit("world_single.png", click=True)
    time.sleep(10)
    board = Battle()
    board.init_reload_interval()
    print(board.get_reload_interval())


class Battle:

    def __init__(self):
        super(Battle, self).__init__()
        self.status = GameStatus.PREPARE_0
        self.reload_interval = 10000
        self.init_reload_interval()

    def get_reload_interval(self):
        return self.reload_interval

    def init_reload_interval(self):
        ocr_result = self.init_reload_interval_async()
        # result formatd
        # [feature1, feature2, ...]
        # For each feature:
        #   feature1 = (Points_list(0),result text(1), confidence(2))
        # For each Points_list:
        #   Points_list = [left-up, right-up, right-down, left-down]
        # For each point in Points_list:
        #   Point = [x,y]
        try:
            reload_time = [int(element[1]) for element in ocr_result if float(element[2]) >= c_confidence]
        except AttributeError as e:
            print("OCR Wrong element")
            return self.reload_interval
        counts = np.bincount(reload_time)
        self.reload_interval = np.argmax(counts)
        return self.reload_interval
        # TODO

    # TODO async design
    def init_reload_interval_async(self) -> list:
        if self.status != GameStatus.PREPARE_0:
            raise ValueError("Calling init out of preparing stage!")
        cropped = cut_image(image=screenshot(), box=RELOAD_BOX)
        return extract_text(image=cropped, num=True)


if __name__ == "__main__":
    print("ADB connecting...")
    os.system(f'adb connect {ip}')
    getResolution()
    print(f'Screen Resolution:{SR}')
    print(f'Click interval: {pausetime}')
    print(f'Confidence setting: {c_confidence}')

    if not os.path.exists(f'./picture/{SR}'):
        print("Unsupported screen resolution")
        os.system('pause')
        os.system(f'adb disconnect {ip}')
        exit(0)
    print("Successful initialization")
    start_game()
    time.sleep(1)
    center_tap()
    time.sleep(pausetime)
    wait_till_screen_static()
    tap(ANNOUNCE_CLOSE)
    time.sleep(pausetime)
    center_tap()
    time.sleep(9)
    locate_or_exit("start_battle.png")
    print("menu found")
    start_1_1()
    # os.system(f'adb disconnect {ip}')
