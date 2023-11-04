import cv2
import numpy as np
import keypoints
import main


def getPiMatrix(Coord2d: np.ndarray, Coord3d: np.ndarray) -> np.ndarray:
    M = createAlgebraicMatrix(Coord2d, Coord3d)
    M = M.T @ M
    eigenvalues, eigenvectors = np.linalg.eig(M)
    smallest_eigenvector = eigenvectors[:, -1]
    P = smallest_eigenvector.reshape((3, 4))
    # Make sure projection matrix has determinant > 0
    if (np.linalg.det(P[:, :3]) < 0):
        P = -P
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


def decomposePiMatrix(P: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    K = np.zeros(shape=(3, 3))
    R = np.zeros(shape=(3, 3))
    u0 = P[0, :3]
    u1 = P[1, :3]
    u2 = P[2, :3]

    K[2, 2] = np.linalg.norm(u2)
    R[2, :] = (u2 / K[2, 2]).reshape((1, 3))

    K[1, 2] = u1 @ R[2, :]
    K[1, 1] = np.linalg.norm(u1 - (K[1, 2] * R[2, :]))
    R[1, :] = (u1 - (K[1, 2] * R[2, :])) / K[1, 1]

    K[0, 2] = u0 @ R[2, :]
    K[0, 1] = u0 @ R[1, :]
    K[0, 0] = np.linalg.norm(u0 - (K[0, 1] * R[1, :]) - (K[0, 2] * R[2, :]))

    R[0, :] = (u0 - (K[0, 1] * R[1, :]) - (K[0, 2] * R[2, :])) / K[0, 0]
    t = np.linalg.inv(K) @ P[:, 3].reshape((3, 1))
    K = K / K[2, 2]
    det = np.linalg.det(R)
    assert 0.5 < det < 1.5
    return K, R, t


def calculate() -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    """
    Calibrates the camera using 2D and 3D coordinates

    :return:
    """
    calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
    calibrationImg = cv2.cvtColor(calibrationImg, cv2.COLOR_BGR2RGB)
    Coord2d, Coord3d = keypoints.calculate(calibrationImg)
    P = getPiMatrix(Coord2d, Coord3d)
    K, R, t = decomposePiMatrix(P)
    return P, K, R, t


if __name__ == "__main__":
    P, K, R, t = calculate()
    main.manageMatrix(
        ["Projection", "Intrinsic", f"Extrinsic Rotation with Determinant: {round(np.linalg.det(R), 5)}",
         "Extrinsic Translation"],
        [P, K, R, t],
        save_result=True,
        file_title="Part 1 Matrices")
