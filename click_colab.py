#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 11:57:53 2024

@author: mrborges
"""

from pynput.mouse import Controller, Button
import time

mouse = Controller()

while True:
    mouse.click(Button.right,1)
    print('clicked')
    time.sleep(30)