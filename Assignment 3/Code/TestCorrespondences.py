import cameracali
import relativepose
import main
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as ptch

configs = ["Distributed", "Edges", "Textured", "Concentrated", "Corners"]
source = cv2.imread("../Assignment 3 Pics/SourceImage.jpg")
source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
target = cv2.imread("../Assignment 3 Pics/TargetImage.jpg")
target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)


def testCorrespondence(c: str) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    P, K, R, t = cameracali.calculate()
    correspondences, ptS, ptT = relativepose.getCustomCorrespondences(c)
    ptS = ptS.T
    ptT = ptT.T
    E = relativepose.estimateEssentialMatrix((P, K, R, t), ptS, ptT)
    R, T = relativepose.getRelativePose(E)
    return E, R, T, ptS, ptT


def saveFig(save: bool = True):
    if save:
        plt.savefig(f"../Output Pictures/{config} Lines.png", dpi=600)


if __name__ == "__main__":
    for config in configs:
        E, R, T, ptS, ptT = testCorrespondence(config)
        fig, (s, t) = plt.subplots(1, 2, figsize=(20, 10))
        s.imshow(source)
        s.set_title(f"Source({config})")
        t.imshow(target)
        t.set_title(f"Target({config})")
        t.set_zorder(-1)
        for i in range(ptS.shape[0]):
            con = ptch.ConnectionPatch(xyA=(ptS[i, 0], ptS[i, 1]), xyB=(ptT[i, 0], ptT[i, 1]), coordsA="data",
                                       coordsB="data", axesA=s, axesB=t, zorder=2, color='b')
            s.add_artist(con)
            saveFig(True)
            main.manageMatrix(
                [f"{config} Essential", f"{config} Rotation with Determinant: {round(np.linalg.det(R), 5)}",
                 f"{config} Translation"],
                [E, R, T],
                save_result=True,
                file_title=f"{config} Matrices", compute=True)
        print(f"{config} Done")
    plt.close("all")
