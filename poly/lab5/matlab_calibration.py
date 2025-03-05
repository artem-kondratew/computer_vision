import cv2 as cv


cap = cv.VideoCapture(2)

a = []

while cap.isOpened():
    _, frame = cap.read()
    cv.imshow('frame', frame)
    key = cv.waitKey(20)
    if key == ord('s') & 0xFF:
        a.append(frame)
    if key == ord('q') & 0xFF:
        break

for i in range(len(a)):
    cv.imwrite(f'frames/frame{i}.jpg', a[i])
