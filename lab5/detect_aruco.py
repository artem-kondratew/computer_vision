import cv2 as cv
import numpy as np


def img_proc():
    aruco5 = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco5, parameters)

    cap = cv.VideoCapture(0)

    while True:
        cap_read_ret, image = cap.read()
        if not cap_read_ret:
            continue

        image = cv.rotate(image, cv.ROTATE_180)
        markerCorners, markerIds, _ = detector.detectMarkers(image)

        cv.aruco.drawDetectedMarkers(image, markerCorners, markerIds)

        cv.imshow('image', image)
        cv.waitKey(20)


if __name__ == '__main__':
    img_proc()
