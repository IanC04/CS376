import keypoints
import cameracali
import cv2
import numpy as np

CalibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
CalibrationImg = cv2.cvtColor(CalibrationImg, cv2.COLOR_BGR2RGB)


def checkMatrices():
    two_d, three_d = keypoints.calculate()
    P = cameracali.getProjectionMatrix(two_d, three_d)
    K, R, t = cameracali.decomposeProjectionMatrix(P)
    intrinsic = K
    extrinsic = np.hstack((R, t))
    while True:
        X = int(input("X: "))
        Y = int(input("Y: "))
        Z = int(input("Z: "))
        imgCoord: np.ndarray = P @ np.array([X, Y, Z, 1])
        imgCoord = imgCoord / imgCoord[-1]
        print(f"({X}, {Y}, {Z}) in world coordinates is ({round(imgCoord[0])}, {round(imgCoord[1])}) in image "
              f"coordinates")
        if input("Continue? (y/n): ") != "y":
            break


def checkKP(two_d: np.ndarray, three_d: np.ndarray, img: np.ndarray = CalibrationImg):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for i in range(len(two_d[0])):
        cv2.circle(img, (two_d[0][i], two_d[1][i]), 5, keypoints.NEON_GREEN, -1)
        cv2.imshow("Image", img)
        print(f"3D box coords: {three_d[0][i], three_d[1][i], three_d[2][i]}")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    # checkKP(*keypoints.calculate(CalibrationImg), CalibrationImg)
    checkMatrices()
