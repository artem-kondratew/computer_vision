import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def construct_filter(pts):
    p0 = np.mean(data, axis=0)
    U, s, Vt = np.linalg.svd(data - p0)
    # print(np.linalg.eigvals(data - p0))
    v = Vt[0, :]
    t = (pts - p0).dot(v)
    print(t[0])
    t1 = np.percentile(t, 5)
    t2 = np.percentile(t, 95)
    dp = np.linalg.norm(np.outer(t, v) + p0 - pts, axis=1)
    R = np.percentile(dp, 95)
    print(R)
    return v, p0, t1, t2, R


def apply_filter(image, v, p0, t1, t2, R):
    p = image.reshape(-1, 3)
    t = (p - p0).dot(v)
    dt = np.abs(t - (t1 + t2) / 2) - (t2 - t1) / 2
    dt = np.maximum(dt, 0)
    dp = np.linalg.norm(np.outer(t, v) + p0 - p, axis=1)
    dp = np.maximum(dp - R, 0)
    d = dp + dt
    return d.reshape(image.shape[:2])


if __name__ == '__main__':
    train = cv.imread('lab2/train.png')
    mask = cv.imread('lab2/mask.png', cv.IMREAD_GRAYSCALE)
    
    h, w = train.shape[:2]
    
    data = train[mask==255]
    
    d = apply_filter(train, *construct_filter(data))
    
    new_mask = np.zeros(train.shape)
    new_mask[d<10] = 255
    
    # size = 10000
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(data[:size, 0], data[:size, 1], data[:size, 2])
    # ax.scatter(mean[0], mean[1], mean[2], marker='^', )
    # ax.grid(True)
    # plt.show()
    
    cv.imshow('train', train)
    cv.imshow('mask', mask)
    cv.imshow('new_mask', new_mask)
    cv.waitKey(0)
    