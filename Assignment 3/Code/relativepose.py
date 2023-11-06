import cv2
import numpy as np

import main
import cameracali

sourceImg = cv2.imread("../Assignment 3 Pics/SourceImage.jpg")
sourceImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2RGB)
targetImg = cv2.imread("../Assignment 3 Pics/TargetImage.jpg")
targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2RGB)


def getRelativePose(E: np.ndarray) -> (np.ndarray, np.ndarray):
    U, S, Vt = np.linalg.svd(E)
    assert abs(S[0] - S[1]) < 0.1 and abs(S[2]) < 0.1
    T = S[0] * U[:, 2]
    T = T.reshape((3, 1))
    R = np.array([-U[:, 1], U[:, 0], U[:, 2]]) @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        T = -T
    return R, T


def estimateEssentialMatrix(matrices: (np.ndarray, np.ndarray, np.ndarray, np.ndarray),
                            ptSource: np.ndarray, ptTarget: np.ndarray) -> (
        np.ndarray):
    E = np.zeros((3, 3))
    P = matrices[0]
    K = matrices[1]
    K_1 = np.linalg.inv(K)
    twoD_Source = np.column_stack((ptSource, np.ones(ptSource.shape[0]))).T
    twoD_Target = np.column_stack((ptTarget, np.ones(ptTarget.shape[0]))).T
    threeD_Source = K_1 @ twoD_Source
    threeD_Target = K_1 @ twoD_Target
    # Get smallest eigenvector corresponding to smallest eigenvalue

    A = np.zeros(shape=(ptSource.shape[0], 9))

    for i in range(ptSource.shape[0]):
        source = threeD_Source[:, i]
        target = threeD_Target[:, i]
        x1, y1, z1 = source[0], source[1], source[2]
        x2, y2, z2 = target[0], target[1], target[2]
        a = np.array([x1 * x2, y1 * x2, z1 * x2, x1 * y2, y1 * y2, z1 * y2, x1 * z2, y1 * z2, z1 * z2]).T
        A[i, :] = a

    A = A.T @ A
    eigenvalues, eigenvectors = np.linalg.eig(A)
    smallest_eigenvector = eigenvectors[:, -1]
    E = smallest_eigenvector.reshape((3, 3))

    # Matrix probably not essential matrix, need to project it onto the essential matrix space
    U, S, Vt = np.linalg.svd(E)
    E = U @ np.diag([1, 1, 0]) @ Vt
    return E


def getCorrespondences() -> (np.ndarray, np.ndarray, np.ndarray):
    sift = cv2.SIFT.create()
    graySource = cv2.cvtColor(sourceImg.copy(), cv2.COLOR_RGB2GRAY)
    grayTarget = cv2.cvtColor(targetImg.copy(), cv2.COLOR_RGB2GRAY)
    kpS, desS = sift.detectAndCompute(graySource, mask=None)
    kpT, desT = sift.detectAndCompute(grayTarget, mask=None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desS, desT)
    matches = sorted(matches, key=lambda x: x.distance)

    matches = np.array(matches)

    # Top 10 matches
    matches = matches[:8]

    ptS = list()
    ptT = list()
    for match in matches:
        ptS.append(kpS[match.queryIdx].pt)
        ptT.append(kpT[match.trainIdx].pt)

    ptS = np.array(ptS)
    ptT = np.array(ptT)

    img_matches = cv2.drawMatches(sourceImg, kpS, targetImg, kpT, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS |
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    main.manageImage(["Matches-SIFT"], [img_matches], display_result=False,
                     save_result=True, file_title="Matches-SIFT", compute=True, dpi=600)
    return matches, ptS, ptT


def calculate() -> (np.ndarray, np.ndarray, np.ndarray):
    P, K, R, t = cameracali.calculate()
    correspondences, ptS, ptT = getCorrespondences()
    E = estimateEssentialMatrix((P, K, R, t), ptS, ptT)
    R, T = getRelativePose(E)
    return E, R, T


def getCustomCorrespondences(config: str) -> (np.ndarray, np.ndarray, np.ndarray):
    correspondences, ptS, ptT = getArgs(config)
    return correspondences, ptS, ptT


def getArgs(c: str) -> (np.ndarray, np.ndarray, np.ndarray):
    corr, source, target = None, None, None
    match c:
        case "Distributed":
            source = np.array(
                [[214, 949, 1900, 2162, 2126, 1799, 1423, 225], [2477, 2657, 1434, 1789, 1886, 684, 1035, 1642]])
            target = np.array(
                [[1568, 3016, 2101, 3220, 3220, 3062, 1760, 279], [2681, 2843, 1394, 1778, 1886, 526, 935, 1602]])
        case "Edges":
            source = np.array(
                [[968, 1303, 1625, 2284, 2161, 2063, 1760, 1500], [2010, 1845, 1706, 1627, 1802, 1424, 1439, 1434]])
            target = np.array(
                [[3067, 3041, 3009, 3225, 2969, 2307, 1893, 1497], [2052, 1846, 1660, 1790, 1594, 1384, 1394, 1376]])
        case "Textured":
            source = np.array(
                [[2042, 1497, 2057, 2738, 1741, 1086, 1265, 2138], [1669, 1691, 1545, 1076, 1569, 1846, 1875, 1889]])
            target = np.array(
                [[2424, 2269, 2277, 2944, 1835, 1279, 2750, 3231], [1645, 1688, 1520, 1053, 1527, 1839, 1882, 1893]])
        case "Concentrated":
            source = np.array(
                [[1938, 1938, 2028, 2028, 1976, 1976, 1945, 2104], [1593, 1581, 1586, 1596, 1540, 1548, 1513, 1455]])
            target = np.array(
                [[2138, 2138, 2247, 2246, 2185, 2185, 2146, 2342], [1565, 1553, 1559, 1570, 1509, 1519, 1480, 1421]])
        case "Corners":
            source = np.array(
                [[215, 1039, 951, 1897, 2261, 1816, 1115, 1406], [2282, 2544, 2649, 1430, 1422, 902, 713, 1676]])
            target = np.array(
                [[1577, 3185, 3023, 2087, 2525, 3104, 3206, 1571], [2423, 2714, 2837, 1397, 1385, 767, 455, 1655]])
        case _:
            raise ValueError("Invalid config")
    return corr, source, target


if __name__ == "__main__":
    E, R, T = calculate()
    main.manageMatrix(
        ["Essential", f"Rotation with Determinant: {round(np.linalg.det(R), 5)}", "Translation"],
        [E, R, T],
        save_result=True,
        file_title="Part 2 Matrices", compute=False)
