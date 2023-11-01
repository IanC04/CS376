import cv2
import numpy as np
import keypoints


def getProjectionMatrix(Coord2d: np.ndarray, Coord3d: np.ndarray) -> np.ndarray:
    A = createAlgebraicMatrix(Coord2d, Coord3d)
    U, S, V = np.linalg.svd(A)
    smallest_eigenvector = V[:, -1]
    P = smallest_eigenvector.reshape((3, 4))
    return P


def createAlgebraicMatrix(imagePoints, worldPoints):
    """
    Create the algebraic matrix A for camera calibration

    :param imagePoints - np.ndarray, shape - (3, points) projections of the above points in the image
    :param worldPoints - np.ndarray, shape - (3, points) points in the world coordinate system
    :return A - np.ndarray, shape - (2 * points, 12) the algebraic matrix used for camera calibration
    """
    assert worldPoints.shape[1] == imagePoints.shape[1]
    points = worldPoints.shape[1]
    A = np.zeros(shape=(2 * points, 12))

    for i in range(points):
        w = worldPoints[:, i]
        p = imagePoints[:, i]

        X, Y, Z = w
        u, v = p
        rows = np.zeros(shape=(2, 12))
        rows[0, 0], rows[0, 1], rows[0, 2], rows[0, 3] = X, Y, Z, 1
        rows[0, 8], rows[0, 9], rows[0, 10], rows[0, 11] = -u * X, -u * Y, -u * Z, -u
        rows[1, 4], rows[1, 5], rows[1, 6], rows[1, 7] = X, Y, Z, 1
        rows[1, 8], rows[1, 9], rows[1, 10], rows[1, 11] = -v * X, -v * Y, -v * Z, -v

        A[2 * i:2 * i + 2, :] = rows
    return A


def decomposeProjectionMatrix(P: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    K, R = np.linalg.qr(P[:, :3])
    t = np.linalg.inv(K) @ P[:, 3].reshape((3, 1))
    return K, R, t


def getExtrinsicMatrix(P: np.ndarray) -> np.ndarray:
    pass


def calculate(Coord2d: np.ndarray, Coord3d: np.ndarray) -> np.ndarray:
    """
    Calibrates the camera using the given 2D and 3D coordinates

    :param Coord2d:
    :param Coord3d:
    :return:
    """
    P = getProjectionMatrix(Coord2d, Coord3d)
    K, R, t = decomposeProjectionMatrix(P)
    return K


if __name__ == "__main__":
    calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
    calibrationImg = cv2.cvtColor(calibrationImg, cv2.COLOR_BGR2RGB)

    two_d, three_d = keypoints.calculate(calibrationImg)
    calculate(two_d, three_d)
