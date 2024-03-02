import cv2 as cv
import numpy as np


aruco5 = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)

tag = np.zeros((300, 300, 1), dtype='uint8')

x = 7
y = 10

w = 2408
h = 3508

s = 300

dx = int((w - x * s) / (x + 1))
dy = int((h - y * s) / (y + 1))

board = np.full((h, w, 1), 255, dtype='uint8')


for j in range(y):
    for i in range(x):
        cv.aruco.generateImageMarker(aruco5, x * j + i, 300, tag, 1)
        px = (i+1) * dx + i * s
        py = (j+1) * dy + j * s
        board[py:py+300, px:px+300] = tag

cv.imwrite('board.png', board)
