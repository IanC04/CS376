import random

import matplotlib.pyplot as plt

import keypoints
import cameracali
import cv2
import numpy as np
import main

CalibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
CalibrationImg = cv2.cvtColor(CalibrationImg, cv2.COLOR_BGR2RGB)


def checkMatrices(P: np.ndarray):
    while True:
        X, Y, Z = (map(int, input("(X, Y, Z): ").split()))
        imgCoord: np.ndarray = P @ np.array([X, Y, Z, 1])
        imgCoord = imgCoord / imgCoord[-1]
        print(f"({X}, {Y}, {Z}) in world coordinates is ({round(imgCoord[0])}, {round(imgCoord[1])}) in image "
              f"coordinates")
        if input("Continue? (y/n): ") != "y":
            break


def checkKP(two_d: np.ndarray, three_d: np.ndarray, P: np.ndarray, img: np.ndarray = CalibrationImg):
    number = len(two_d[0])
    img_copy = img.copy()
    for j in range(number):
        cv2.circle(img_copy, (two_d[0][j], two_d[1][j]), 6, keypoints.NEON_GREEN, -1)
        X, Y, Z = three_d[0][j], three_d[1][j], three_d[2][j]
        imgCoord: np.ndarray = P @ np.array([X, Y, Z, 1])
        imgCoord = imgCoord / imgCoord[-1]
        cv2.circle(img_copy, (round(imgCoord[0]), round(imgCoord[1])), 6, keypoints.NEON_BLUE, -1)
        print(f"3D box coords: {three_d[0][j], three_d[1][j], three_d[2][j]}")

    for k in range(10):
        imgCoord: np.ndarray = (P @ np.array([random.randint(0, 5), random.randint(0, 5),
                                              random.randint(0, 3), 1]))
        imgCoord = imgCoord / imgCoord[-1]
        cv2.circle(img_copy, (round(imgCoord[0]), round(imgCoord[1])), 6, keypoints.NEON_RED, -1)
    return img_copy


if __name__ == "__main__":
    imgs = list()
    Ps = list()
    for i in range(5):
        two_d, three_d = keypoints.tryConfiguration(i)
        P = cameracali.getPiMatrix(two_d, three_d)
        Ps.append(P)
        K, R, t = cameracali.decomposePiMatrix(P)
        if K is None or R is None or t is None:
            raise Exception("None matrix")
        intrinsic = K
        extrinsic = np.hstack((R, t))
        print(f"Configuration {i}")
        img = checkKP(two_d, three_d, P)
        imgs.append(img)

    main.manageImage(["Original", "Reduced original keypoints", "Clustered near Origin", "Spread out",
                      "Fewer keypoints"], imgs,
                     save_result=True,
                     file_title="Configurations",
                     legend_data="Green: Picked Keypoints\nBlue: Recalculated 3D Coordinates\nRed: Random "
                                 "Intersection Points",
                     compute=False)

    imgs = list()
    perm = np.array(np.meshgrid(np.arange(0, 9), np.arange(0, 9), np.arange(0, 5))).T.reshape(-1, 3)
    for i in range(len(Ps)):
        imgs.append(CalibrationImg.copy())
        img = imgs[i]
        for j in range(perm.shape[0]):
            imgCoord: np.ndarray = (Ps[i] @ np.array([perm[j][0], perm[j][1], perm[j][2], 1]))
            imgCoord = imgCoord / imgCoord[-1]
            cv2.circle(img, (round(imgCoord[0]), round(imgCoord[1])), 6, keypoints.NEON_RED, -1)
    main.manageImage(["Original", "Reduced original keypoints", "Clustered near Origin", "Spread out",
                      "Fewer keypoints"], imgs,
                     save_result=True,
                     file_title="Reconstructed Boxes", compute=False)
    # checkMatrices(P)
