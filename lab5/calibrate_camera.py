import cv2 as cv


nx = 7
ny = 10
l = 0.025 # m
d = 0.004 # m
aruco_dict = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_5X5_250)
output_file = './calibration_params.yaml'


def main():
    cap = cv.VideoCapture(0)

    all_marker_corners = []
    all_marker_ids = []

    parameters = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(aruco_dict, parameters)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        marker_corners, marker_ids, _ = detector.detectMarkers(frame)

        if(marker_corners):
            cv.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)

        cv.imshow('frame', frame)

        key = cv.waitKey(20)

        if key == 27:
             break

        if key == ord('c') & 0xFF and marker_corners:
            print('frame captured')
            all_marker_corners.append(marker_corners)
            all_marker_ids.append(marker_ids)
            h, w = frame.shape[:2]

    n_frames = len(all_marker_corners)
    gridboard = cv.aruco.GridBoard((nx, ny), l, d, aruco_dict)

    processed_image_points = []
    processed_object_points = []

    for frame in range(n_frames):
        current_obj_points, current_img_points = gridboard.matchImagePoints(\
            all_marker_corners[frame], all_marker_ids[frame])
        if len(current_img_points) > 0 and len(current_obj_points) > 0:
            processed_image_points.append(current_img_points)
            processed_object_points.append(current_obj_points)

    mtx = None
    dist = None
    _, mtx, dist, _, _ = cv.calibrateCamera(processed_object_points, \
                                            processed_image_points, (w, h), mtx, dist)
    fs = cv.FileStorage(output_file, flags=1)
    if not fs.isOpened():
        print(f'cannot open {output_file}')
    fs.write('width', w)
    fs.write('height', h)
    fs.write('mtx', mtx)
    fs.write('dist', dist)

    print(f'saved to {output_file}')

if __name__ == '__main__':
    main()


