import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def readImagesAndTimes(paths):
    image = []
    for x in paths:
        image.append(cv.imread(x))
    align = cv.createAlignMTB()
    align.process(image, image)
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
    result_image = np.clip(HDR_result * 255, 0, 255).astype('uint8')
    cv.imwrite("./Test Images/2/Inconsistent/Result.jpg", result_image)

def graph_camera_response_function(images, times):
    calibration = cv.createCalibrateDebevec()
    crf_debevec = calibration.process(images, times=times)
    return crf_debevec

if __name__ == "__main__":
    time = np.array([1.0/142.0, 1.0/91.0, 1.0/56.0, 1.0/30.0], dtype=np.float32)
    paths = ["./Test Images/5/Inconsistent/1.jpeg",
             "./Test Images/5/Inconsistent/2.jpeg",
             "./Test Images/5/Inconsistent/3.jpeg",
             "./Test Images/5/Inconsistent/4.jpeg"]
    images = readImagesAndTimes(paths)
    merge_mertens = cv.createMergeMertens()
    fusion = merge_mertens.process(images)
    fusion_8bit = np.clip(fusion * 255, 0, 255).astype('uint8')
    cv.imwrite("./Test Images/5/Inconsistent/fusion_result.png", fusion_8bit)
    hdr_merge = tone_merge(images, time)
    mapping = tone_map(hdr_merge)
    save_result(mapping)
    x_value = [x for x in range(0, 256)]
    vectors = graph_camera_response_function(images, time)
    plt.plot(x_value, vectors[:, :, 0], label='B', color='b')
    plt.plot(x_value, vectors[:, :, 1], label='G', color='g')
    plt.plot(x_value, vectors[:, :, 2], label='R', color='r')
    plt.xlabel("pixel value intensity")
    plt.ylabel("Calibrated intensity")
    plt.legend()
    plt.title("Estimated camera response function for consistent white balance")
    plt.savefig('./Test Images/5/Inconsistent/CRF.png')
    plt.show()
