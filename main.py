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
c_confidence = config.get('main', 'Confidence')
SC = config.getboolean('main', 'StoneCrush')


def adb_send(command):
    os.system(f"adb -s {ip} {command}")


def screenshot(id=0):
    adb_send(f"shell mkdir -p {Apath}")
    adb_send(f"shell screencap -p {Apath}/{id}_{filename}")
    adb_send(f"pull {Apath}/{id}_{filename}")


def get_filename_by_id(id=0):
    return f"{id}_{filename}"


def getResolution():
    screenshot(0)
    try:
        global SR  # ScreenResolution
        global RES
        print(get_filename_by_id(0))
        img = Image.open(get_filename_by_id(0))
        SR = f'{img.size[0]}x{img.size[1]}'
        RES = (img.size[0], img.size[1])
        return SR
    except:
        print('ADB Connection error')
        sys.exit(0)


def center_tap():
    tap(RES[0] // 2, RES[1] // 2)


def cut_image(box):
    utility.cut_image_internal(filename=get_filename_by_id(0), outname=f"./picture/cropped/textarea.png", box=box)


def extract_text(num=False):
    return utility.extract_text_internal(filename=f"./picture/cropped/textarea.png", num=num)


def tap(x: int or tuple, y=None):
    if y is None:
        x, y = x
    print(f"Tap event at point {x} {y}")
    adb_send(f"shell input tap {x} {y}")


def locatePic(pic: str, msg=None, loop=1, pause=pausetime, click=True, id=0):
    p = None
    while loop > 0:
        p = auto.locate(f"./picture/{SR}/{pic}", get_filename_by_id(id), confidence=0.8)
        if p:
            if msg:
                print(msg)
            if click:
                tap(auto.center(p))
                time.sleep(pause)
            break
        loop -= 1
    return p


def screen_change(pause=pausetime):
    screenshot(0)
    time.sleep(pause)
    screenshot(1)
    im1 = cv2.imread(get_filename_by_id(0))
    im2 = cv2.imread(get_filename_by_id(1))
    if im1.shape != im2.shape:
        raise ValueError("Change of device resolution or orientation.")
    difference = cv2.subtract(im1, im2)
    result = np.any(difference)
    return result


def locate_or_exit(pic, click=False):
    p = None
    for attempt in range(RECOGNITION_ATTEMPTS):
        screenshot(0)
        p = locatePic(pic, loop=RECOGNITION_ATTEMPTS, click=click, id=0)
        if p is not None:
            return p
        time.sleep(pausetime)
    if p is None:
        raise ValueError("Recognition failure")
    return p


def wait_till_screen_static(timeout=5):
    sum = 0
    while sum < timeout:
        if not screen_change(pausetime):
            break
        sum += pausetime


def start_game():
    locatePic("NC_APPIcon.png")


def start_1_1():
    tap(WORLD)
    time.sleep(QUICK_PAUSE)
    tap(WORLD_1)
    time.sleep(QUICK_PAUSE)
    locate_or_exit("1-1.png", click=True)
    time.sleep(QUICK_PAUSE)
    locate_or_exit("world_single.png", click=True)
    time.sleep(8)
    board = Battle()
    board.get_reload_interval()


class Battle:

    def __init__(self):
        super(Battle, self).__init__()
        self.status = GameStatus.PREPARE_0
        self.reload_interval = 10000

    def get_reload_interval(self):
        return self.get_reload_interval()

    def init_reload_interval(self):
        ocr_result = asyncio.wait_for(self.init_reload_interval_async(), timeout=OCR_TIME_OUT, loop=None)


    async def init_reload_interval_async(self):
        if self.status != GameStatus.PREPARE_0:
            raise ValueError("Calling init out of preparing stage!")
        screenshot(0)
        cut_image(RELOAD_BOX)
        return extract_text(num=True)


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
    time.sleep(0.5)
    center_tap()
    time.sleep(0.5)
    wait_till_screen_static()
    tap(ANNOUNCE_CLOSE)
    time.sleep(pausetime)
    center_tap()
    time.sleep(5)
    locate_or_exit("start_battle.png")
    start_1_1()
    # os.system(f'adb disconnect {ip}')
