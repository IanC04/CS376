import cv2
import numpy as np

import main
import cameracali

sourceImg = cv2.imread("../Assignment 3 Pics/SourceImage.jpg")
sourceImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2RGB)
targetImg = cv2.imread("../Assignment 3 Pics/TargetImage.jpg")
targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2RGB)


def getRelativePose(E: np.ndarray) -> (np.ndarray, np.ndarray):
    # TODO
    return E, E


def estimateEssentialMatrix(matrices: (np.ndarray, np.ndarray, np.ndarray, np.ndarray), correspondences: np.ndarray,
                            ptSource: np.ndarray, ptTarget: np.ndarray) -> (
        np.ndarray):
    P = matrices[0]
    K = matrices[1]
    threeD_Source = P @ ptSource
    threeD_Target = P @ ptTarget
    x1 = correspondences[:, 0]
    # TODO use K matrix(inverse?) to get 3d coordinates then get smallest eigenvector corresponding to smallest
    #  eigenvalue
    return correspondences


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
    matches = matches[:10]

    ptS = list()
    ptT = list()
    for match in matches:
        ptS.append(kpS[match.queryIdx].pt)
        ptT.append(kpT[match.trainIdx].pt)

    ptS = np.array(ptS)
    ptT = np.array(ptT)

    img_matches = cv2.drawMatches(sourceImg, kpS, targetImg, kpT, matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    main.manageImage(["Matches-SIFT"], [img_matches], display_result=False,
                     save_result=False, file_title="Matches-SIFT")
    return matches, ptS, ptT


def calculate() -> (np.ndarray, np.ndarray):
    P, K, R, t = cameracali.calculate()
    correspondences, ptS, ptT = getCorrespondences()
    E = estimateEssentialMatrix((P, K, R, t), correspondences, ptS, ptT)
    R, T = getRelativePose(E)
    return R, T


if __name__ == "__main__":
    calculate()
