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
    # TODO make sure S[0] == S[1] and S[2] == 0
    T = S[0] * U[:, 2]
    T = T.reshape((3, 1))
    R = np.array([-U[:, 1], U[:, 0], U[:, 2]]) @ Vt
    if np.linalg.det(R) < 0:
        R = -R
        T = -T
    return R, T


def estimateEssentialMatrix(matrices: (np.ndarray, np.ndarray, np.ndarray, np.ndarray), correspondences: np.ndarray,
                            ptSource: np.ndarray, ptTarget: np.ndarray) -> (
        np.ndarray):
    E = np.zeros((3, 3))
    P = matrices[0]
    K = matrices[1]
    # TODO: fix (X, Y, Z) = K^-1 (x, y, 1)
    K_1 = np.linalg.inv(K)
    twoD_Source = np.column_stack((ptSource, np.ones(ptSource.shape[0]))).T
    twoD_Target = np.column_stack((ptTarget, np.ones(ptTarget.shape[0]))).T
    threeD_Source = K_1 @ twoD_Source
    threeD_Target = K_1 @ twoD_Target
    # Get smallest eigenvector corresponding to smallest eigenvalue

    A = np.zeros(shape=(correspondences.shape[0], 9))

    for i in range(correspondences.shape[0]):
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
                     save_result=False, file_title="Matches-SIFT", compute=False)
    return matches, ptS, ptT


def calculate() -> (np.ndarray, np.ndarray, np.ndarray):
    P, K, R, t = cameracali.calculate()
    correspondences, ptS, ptT = getCorrespondences()
    E = estimateEssentialMatrix((P, K, R, t), correspondences, ptS, ptT)
    R, T = getRelativePose(E)
    return E, R, T


if __name__ == "__main__":
    E, R, T = calculate()
    main.manageMatrix(
        ["Essential", f"Rotation with Determinant: {round(np.linalg.det(R), 5)}", "Translation"],
        [E, R, T],
        save_result=True,
        file_title="Part 2 Matrices", compute=False)
