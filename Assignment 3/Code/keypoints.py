import cv2
import numpy as np

calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
calibrationImg = cv2.cvtColor(calibrationImg, cv2.COLOR_BGR2RGB)
NEON_GREEN = (57, 255, 0)
NEON_BLUE = (31, 81, 255)
NEON_RED = (255, 49, 49)


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


def tryConfiguration(i: int) -> (np.ndarray, np.ndarray):
    match i:
        case 0:
            return (np.array([[1971, 1377], [1441, 1558], [1706, 1559], [1971, 1555], [2236, 1553], [2499, 1553],
                              [1149, 1762], [2765, 1742], [1504, 2112], [1975, 2750]], dtype=np.int64).T,
                    np.array([[6, 6, 0], [5, 1, 0], [4, 2, 0], [3, 3, 0], [2, 4, 0], [1, 5, 0],
                              [6, 0, 1], [0, 6, 1], [3, 0, 2], [0, 0, 4]], dtype=np.int64).T)

        case 1:
            return (np.array([[1971, 1377], [1441, 1558], [1706, 1559], [1824, 1685], [2234, 1554], [2765, 1742]],
                             dtype=np.int64).T,
                    np.array([[6, 6, 0], [5, 1, 0], [4, 2, 0], [2, 1, 0], [1, 4, 0], [0, 6, 1]], dtype=np.int64).T)

        case 2:
            return (np.array([[1971, 1866], [1804, 1806], [1975, 1735], [2139, 1791], [2142, 2016], [1781, 2040]],
                             dtype=np.int64).T,
                    np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.int64).T)

        case 3:
            return (np.array([[962, 2010], [1969, 1862], [1974, 2748], [2935, 1981], [2863, 1526], [2160, 1293]],
                             dtype=np.int64).T,
                    np.array([[8, 0, 3], [0, 0, 0], [0, 0, 4], [0, 8, 3], [0, 7, 0], [7, 9, 0]], dtype=np.int64).T)

        case 4:
            return (np.array([[1972, 1425], [2650, 2146], [1250, 1993], [1975, 2543], [1970, 1863]],
                             dtype=np.int64).T,
                    np.array([[5, 5, 0], [0, 5, 3], [5, 0, 2], [0, 0, 3], [0, 0, 0]], dtype=np.int64).T)
    pass


def getAllKeypoints() -> (np.ndarray, np.ndarray):
    two_d = np.array([[1971, 1555], [2236, 1553], [2499, 1553],
                      [2234, 1554], [1975, 2750], [1971, 1866],
                      [1804, 1806], [1975, 1735], [2139, 1791], [2142, 2016], [1781, 2040], [962, 2010], [2935, 1981],
                      [2863, 1526], [2160, 1293], [1972, 1425], [2650, 2146], [1250, 1993],
                      [1975, 2543]], dtype=np.int64).T

    three_d = np.array([[3, 3, 0], [2, 4, 0], [1, 5, 0],
                        [1, 4, 0], [0, 0, 4], [0, 0, 0], [1, 0, 0],
                        [1, 1, 0],
                        [0, 1, 0], [0, 1, 1], [1, 0, 1], [8, 0, 3], [0, 8, 3], [0, 7, 0],
                        [7, 9, 0],
                        [5, 5, 0], [0, 5, 3], [5, 0, 2], [0, 0, 3]], dtype=np.int64).T
    return two_d, three_d


if __name__ == "__main__":
    calculate()
