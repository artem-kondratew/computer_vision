import cv2 as cv
import numpy as np


def get_bw_transition(data, x, y):
    idc = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1))
    cnt = 0
    val = data[y, x]
    for idx in idc:
        new_val = data[y+idx[1], x+idx[0]]
        if val == 0 and new_val == 255:
            cnt += 1
        val = new_val
    return cnt


def get_white_nbs(data, x, y):
    cnt = 0
    for i in range(y - 1, y + 2):
        for j in range(x - 1, x + 2):
            if i == y and j == x:
                continue
            if data[i, j] == 255:
                cnt += 1
    return cnt


def check_246(data, x, y):
    if data[y+1, x] == 0 or data[y, x+1] == 0 or data[y-1, x] == 0:
        return True
    return False


def check_468(data, x, y):
    if data[y, x+1] == 0 or data[y-1, x] == 0 or data[y, x-1] == 0:
        return True
    return False


def check_248(data, x, y):
    if data[y+1, x] == 0 or data[y, x+1] == 0 or data[y, x-1] == 0:
        return True
    return False


def check_268(data, x, y):
    if data[y+1, x] == 0 or data[y-1, x] == 0 or data[y, x-1] == 0:
        return True
    return False


def skeleton_0(data):
    skeleton = np.zeros(data.shape, dtype='uint8')
    h, w = data.shape[:2]
    s = ''
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if s == 'qq':
                print('ERROR')
                exit(1)
            s = ''
            if data[y, x] == 255:
                if 2 <= get_white_nbs(data, x, y) <= 6 and get_bw_transition(data, x, y) == 1:
                    if check_246(data, x, y) and check_468(data, x, y):
                        s += 'q'
                        continue
            skeleton[y, x] = data[y, x]
            s += 'q'
    return skeleton


def skeleton_1(data):
    skeleton = np.zeros(data.shape, dtype='uint8')
    h, w = data.shape[:2]
    s = ''
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if s == 'qq':
                print('ERROR')
                exit(1)
            s = ''
            if data[y, x] == 255:
                if 2 <= get_white_nbs(data, x, y) <= 6 and get_bw_transition(data, x, y) == 1:
                    if check_248(data, x, y) and check_268(data, x, y):
                        s += 'q'
                        continue
            skeleton[y, x] = data[y, x]
            s += 'q'
    return skeleton


def skeleton(binary):
    skeleton = binary.copy()
    flag = True
    cnt = 0
    while(flag):
        print(cnt)
        new_skeleton = skeleton_1(skeleton_0(skeleton))
        if np.array_equal(skeleton, new_skeleton):
            flag = False
        skeleton = new_skeleton
        cnt += 1
    return skeleton
