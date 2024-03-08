import cv2 as cv
import numpy as np


aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
aruco_marker_size = 0.2  # m
d = aruco_marker_size
mtx = None
dist = None
detector = None


def find_recs_tvecs(corners, marker_sz, cap_mtx, cap_dist):
    marker_points = np.array([[-marker_sz / 2, marker_sz / 2, 0],
                              [marker_sz / 2, marker_sz / 2, 0],
                              [marker_sz / 2, -marker_sz / 2, 0],
                              [-marker_sz / 2, -marker_sz / 2, 0]], dtype=np.float32)
    _, rvec, tvec = cv.solvePnP(marker_points, corners[0], cap_mtx, cap_dist, \
                                False, cv.SOLVEPNP_IPPE_SQUARE)
    return rvec, tvec


def draw(frame):
    marker_corners, _, _ = detector.detectMarkers(frame)
    if not marker_corners:
        return frame
    rvec, tvec = find_recs_tvecs(marker_corners, aruco_marker_size, mtx, dist)

    pts = np.array([[+d/2, +d/2, 0],
                    [-d/2, +d/2, 0],
                    [-d/2, -d/2, 0],
                    [+d/2, -d/2, 0],
                    [+d/2, +d/2, d],
                    [-d/2, +d/2, d],
                    [-d/2, -d/2, d],
                    [+d/2, -d/2, d]])
    
    frame_pts, _ = cv.projectPoints(pts, rvec, tvec, mtx, dist)
    correct_pts = []
    for pt in frame_pts:
        pt = pt[0]
        pt = (int(pt[0]), int(pt[1]))
        correct_pts.append(pt)
        cv.circle(frame, pt, 4, (255, 0, 255), 1)

    blue = (255, 0, 0)
    green = (0, 255, 0)
    red = (0, 0, 255)
    
    cv.line(frame, correct_pts[0], correct_pts[1], green, 1)
    cv.line(frame, correct_pts[2], correct_pts[3], green, 1)
    cv.line(frame, correct_pts[4], correct_pts[5], green, 1)
    cv.line(frame, correct_pts[6], correct_pts[7], green, 1)
    cv.line(frame, correct_pts[0], correct_pts[4], blue, 1)
    cv.line(frame, correct_pts[1], correct_pts[5], blue, 1)
    cv.line(frame, correct_pts[2], correct_pts[6], blue, 1)
    cv.line(frame, correct_pts[3], correct_pts[7], blue, 1)
    cv.line(frame, correct_pts[0], correct_pts[3], red, 1)
    cv.line(frame, correct_pts[1], correct_pts[2], red, 1)
    cv.line(frame, correct_pts[4], correct_pts[7], red, 1)
    cv.line(frame, correct_pts[5], correct_pts[6], red, 1)

    return frame


def main():
    global mtx, dist, detector
    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)
    path = '/home/user/projects/computer_vision/lab5/'
    output_file = path + 'calibration_params.yaml'
    fs = cv.FileStorage(output_file, cv.FileStorage_READ)
    if not fs.isOpened():
        print(f'cannot open {output_file}')
    mtx = fs.getNode('mtx').mat()
    dist = fs.getNode('dist').mat()

    cap = cv.VideoCapture(2)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
        frame = draw(frame)
        frame = cv.resize(frame, (640, 480))
        cv.imshow('window', frame)
        cv.waitKey(20)


if __name__ == '__main__':
    main()
    if ord('q') & 0xFF == cv.waitKey(0):
        exit(0)
