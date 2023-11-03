import cv2
import numpy as np

calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
calibrationImg = cv2.cvtColor(calibrationImg, cv2.COLOR_BGR2RGB)
NEON_GREEN = (57, 255, 0)


def get2D(img: np.ndarray) -> np.ndarray:
    """
    Get the coordinates of the keypoints in (x, y) starting from the top-left corner

    :param img: Image to get the keypoints from
    :return:
    """
    coords2D = np.array([[1971, 1377], [1441, 1558], [1706, 1559], [1971, 1555], [2236, 1553], [2499, 1553],
                         [1149, 1762], [2765, 1742], [1504, 2112], [1975, 2750]], dtype=np.int64).T
    return coords2D


def get3D(img: np.ndarray) -> np.ndarray:
    """
    Get the coordinates of the keypoints in (x, y, z) starting from the front middle of the calibration rig
    x points left, y points right, z points down
    :param img:
    :return:
    """
    coords3D = np.array([[6, 6, 0], [5, 1, 0], [4, 2, 0], [3, 3, 0], [2, 4, 0], [1, 5, 0],
                         [6, 0, 1], [0, 6, 1], [3, 0, 2], [0, 0, 4]], dtype=np.int64).T
    return coords3D


def calculate(img: np.ndarray = calibrationImg) -> (np.ndarray, np.ndarray):
    """
    Calculates the 2D and 3D coordinates of the keypoints in the image

    :param img:
    :return:
    """
    two_d = get2D(img)
    three_d = get3D(img)
    return two_d, three_d


if __name__ == "__main__":
    calculate()
