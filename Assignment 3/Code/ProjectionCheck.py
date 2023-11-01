import keypoints
import cameracali
import numpy as np

if __name__ == "__main__":
    two_d, three_d = keypoints.calculate()
    P = cameracali.getProjectionMatrix(two_d, three_d)
    while input("Continue? (y/n): ") == "y":
        X = int(input("X: "))
        Y = int(input("Y: "))
        Z = int(input("Z: "))
        imgCoord: np.ndarray = P @ np.array([X, Y, Z, 1])
        imgCoord = imgCoord / imgCoord[-1]
        print(f"({X}, {Y}, {Z}) in world coordinates is ({round(imgCoord[0])}, {round(imgCoord[1])}) in image "
              f"coordinates")