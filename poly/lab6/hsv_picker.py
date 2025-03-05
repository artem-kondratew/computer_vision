import cv2 as cv
import numpy as np


def nothing(x):
    return x


if __name__ == "__main__":
    width = 640
    height = 480

    cap = cv.VideoCapture('calibration/calib2.avi')

    # Load image
    image = cv.imread('calibration/calib0.jpg')

    # Create a window
    cv.namedWindow('image')

    # Create trackbars for color change
    # Hue is from 0-179 for Opencv
    cv.createTrackbar('HMin', 'image', 0, 179, nothing)
    cv.createTrackbar('SMin', 'image', 0, 255, nothing)
    cv.createTrackbar('VMin', 'image', 0, 255, nothing)
    cv.createTrackbar('HMax', 'image', 0, 179, nothing)
    cv.createTrackbar('SMax', 'image', 0, 255, nothing)
    cv.createTrackbar('VMax', 'image', 0, 255, nothing)

    # Set default value for Max HSV trackbars
    cv.setTrackbarPos('HMax', 'image', 179)
    cv.setTrackbarPos('SMax', 'image', 255)
    cv.setTrackbarPos('VMax', 'image', 255)

    # Initialize HSV min/max values
    hMin = sMin = vMin = hMax = sMax = vMax = 0
    phMin = psMin = pvMin = phMax = psMax = pvMax = 0

    while True:
        # ret, image = cap.read()
        # if not ret:
        #     continue

        # Get current positions of all trackbars
        hMin = cv.getTrackbarPos('HMin', 'image')
        sMin = cv.getTrackbarPos('SMin', 'image')
        hMax = cv.getTrackbarPos('HMax', 'image')
        vMin = cv.getTrackbarPos('VMin', 'image')
        sMax = cv.getTrackbarPos('SMax', 'image')
        vMax = cv.getTrackbarPos('VMax', 'image')

        # Set minimum and maximum HSV values to display
        lower = np.array([hMin, sMin, vMin])
        upper = np.array([hMax, sMax, vMax])

        # Convert to HSV format and color threshold
        hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(image, image, mask=mask)

        # Print if there is a change in HSV value
        if (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax):
            phMin = hMin
            psMin = sMin
            pvMin = vMin
            phMax = hMax
            psMax = sMax
            pvMax = vMax

        # Display result image
        cv.imshow('image', result)
        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    minimum = (phMin, psMin, pvMin)
    maximum = (phMax, psMax, pvMax)
    print("min: ", minimum)
    print("max: ", maximum)

    cv.destroyAllWindows()