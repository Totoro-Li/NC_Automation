#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Filename : exceptions
# @Date : 2022-07-30
# @Project: NC_Automation
# @AUTHOR : Totoro
__all__ = [
    "GameStatusError"
]


class GameStatusError(Exception):
    def __init__(self, ErrorInfo):
        super().__init__(self)
        self.errorinfo = ErrorInfo

    def __str__(self):
        return self.errorinfo
