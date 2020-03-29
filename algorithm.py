import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def readImagesAndTimes(paths):
    image = []
    for x in paths:
        image.append(cv.imread(x))
    return image

def tone_merge(images, times):
    merge = cv.createMergeDebevec()
    hdr = merge.process(images, times=times.copy())
    return hdr

def tone_map(HDR_object):
    tonemap = cv.createTonemap(gamma=2.2)
    result = tonemap.process(HDR_object.copy())
    return result

def save_result(HDR_result):
    result_image = np.clip(HDR_result * 255, 0, 255).astype('uint16')
    cv.imwrite("Result.jpg", result_image)

def graph_camera_response_function(images, times):
    calibration = cv.createCalibrateDebevec()
    crf_debevec = calibration.process(images, times=times)
    return crf_debevec

if __name__ == "__main__":
    time = np.array([1/50, 1/4, 1, 4], dtype=np.float32)
    paths = ["./Test Images/3/111.jpg",
             "./Test Images/3/222.jpg",
             "./Test Images/3/333.jpg",
             "./Test Images/3/444.jpg"]
    images = readImagesAndTimes(paths)
    hdr_merge = tone_merge(images, time)
    mapping = tone_map(hdr_merge)
    save_result(mapping)
    x_value = [x for x in range(0, 256)]
    vectors = graph_camera_response_function(images, time)
    plt.plot(x_value, vectors[:, :, 0])
    plt.plot(x_value, vectors[:, :, 1])
    plt.plot(x_value, vectors[:, :, 2])
    plt.savefig('3.png')
    plt.show()
