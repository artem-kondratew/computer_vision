import cv2 as cv
import numpy as np


def main():
    frame = cv.imread('calibration/calib0.jpg')

    min_hsv = (0, 0, 120)
    max_hsv = (94, 112, 255)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    laser = cv.inRange(hsv, min_hsv, max_hsv)
    h, w = frame.shape[:2]
    cx, cy = (w // 2, h // 2)
    aovx = 74 * np.pi / 180
    aovy = 2 * np.arctan(np.tan(aovx / 2) * h / w)
    fx = w / 2 / np.tan(aovx / 2)
    fy = h / 2 / np.tan(aovy / 2)

    map2d = np.zeros((480, 640, 3), dtype='uint8')

    for i in range(map2d.shape[0]):
        for j in range(map2d.shape[1]):
            if (i % 20 == 0 or j % 20 == 0):
                map2d[i, j] = (255, 255, 255)

    for i in range(h):
        for j in range(w):
            if laser[i, j] == 0:
                continue
            ax = (j - cx) / fx
            ay = (i - cy) / fy
            x = int(map2d.shape[1] / 2 + 20 * np.tan(ax) / np.tan(ay))
            y = int(map2d.shape[0] / 2 - 20 / np.tan(ay))
            cv.circle(map2d, (x, y), 1, (0, 255, 0), -1)

    cv.imshow('camera', frame)
    cv.imshow('laser', laser)
    cv.imshow('map', map2d)
    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
