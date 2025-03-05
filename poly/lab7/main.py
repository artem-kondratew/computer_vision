import cv2 as cv
import numpy as np

from scipy.spatial import distance

from skeleton import skeleton as sk


def main():
    image = cv.imread('./data/image2.jpg')
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 90, 255, cv.THRESH_BINARY)

    skeleton = sk(binary)
    
    lines = np.zeros(skeleton.shape, dtype='uint8')
    lines = cv.cvtColor(lines, cv.COLOR_GRAY2BGR)
    contours, _ = cv.findContours(skeleton, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(lines, contours, -1, (255, 0, 255), 1)

    points = []
    for contour in contours:
        if len(contour) < 12:
            continue
        pts = contour.squeeze(axis=1)
        for pt in pts:
            x, y = pt
            nb = skeleton[y-1:y+2, x-1:x+2]
            if np.sum(nb) == 255 * 2:
                points.append(pt)
                cv.circle(lines, (x, y), 3, (255, 0, 255), 1)

    points.sort(key = lambda a: a[1])
    print(points)
    a = [True for i in range(len(points))]

    for i in range(len(points)):
        print(points[i], end=' ')
        min_dist = float('inf')
        min_idx = None
        for j in range(len(points)):
            if i == j or a[j] == False:
                continue
            dist = distance.euclidean(points[i], points[j])
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        if min_idx == None:
            continue
        cv.line(lines, points[i], points[min_idx], (255, 0, 0), 1)
        print(points[min_idx])
        cv.imshow('lines', lines)
        cv.waitKey(0)
        a[i] = False

    cv.imshow('image', image)
    cv.imshow('binary', binary)
    cv.imshow('skeleton', skeleton)
    cv.imshow('lines', lines)

    cv.waitKey(0)


if __name__ == '__main__':
    main()
