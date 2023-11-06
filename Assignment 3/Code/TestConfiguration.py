import keypoints
import cameracali
import numpy as np
import main
import cv2
from itertools import combinations

Calibration_img = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
Calibration_img = cv2.cvtColor(Calibration_img, cv2.COLOR_BGR2RGB)

if __name__ == "__main__":
    two_d, three_d = keypoints.getAllKeypoints()
    associated = np.row_stack((two_d, three_d)).T
    associated = associated.tolist()
    combs = combinations(associated, r=10)
    best_ones = list()
    for i in combs:
        j = np.array(i).T
        two_d, three_d = j[:2], j[2:]
        P = cameracali.getPiMatrix(two_d, three_d)
        K, R, t = cameracali.decomposePiMatrix(P)
        if K is None or R is None or t is None:
            continue
        imgCoords = np.zeros((3, len(three_d[0])))
        score = 0
        for j in range(len(three_d[0])):
            imgCoord = P @ np.array([three_d[0, j], three_d[1, j], three_d[2, j], 1])
            imgCoords[:, j] = imgCoord / imgCoord[-1]
            score += np.linalg.norm(two_d[:, j] - imgCoords[:-1, j])
        best_ones.append((two_d, score))
    best_ones.sort(key=lambda x: x[1])
    imgs = [Calibration_img.copy(), Calibration_img.copy(), Calibration_img.copy()]
    for i in range(3):
        for j in range(10):
            cv2.circle(imgs[i], (best_ones[i][0][0][j], best_ones[i][0][1][j]), 6, keypoints.NEON_GREEN, -1)
    main.manageImage(["1st Place", "2nd Place", "3rd Place"], imgs, save_result=True, file_title="Best "
                                                                                                 "Configurations",
                     compute=True)
