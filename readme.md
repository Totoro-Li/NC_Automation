# Automation script for Naval Creed based on OpenCV and ADB tools

## Description

Python frame parser for Naval Creed that supprts parsing of legend, interface and basic operations.
For convenience of ADB connection and unifying of resolution, all assets and functions are only tested on the basis of  Bluestacks 5.
Supported resolution:

- 2500x1080(with default layout)

## Python env setting up

All dependencies are tested OK on Python 3.10, Windows 10(x64), CUDA 11.6(with Nvidia RTX3060 Laptop)

Method 1：Auto configuration，run script `env.bat` (with Python, pip and venv installed)

Method 2: Manual，install dependencies in requirements.txt

```bash
pip install -r requirements.txt
```

## Usage

Create new configuration file config.ini

```ini
[main]

; Operation interval
PauseTime=1
; ADB port of your Android emulator
IP="localhost:5555"
; Confidence threshold of OCR and feature recognition
Confidence=0.8
```

## Changelog

- Beta 0.1 Basic implementation of OCR and frame parsing.

## Contributor
- Z.Li
- Awaiting more NC players to join and refine this project that is still on her initial stage of development .