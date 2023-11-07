import main
import cameracali
import cv2
import numpy as np

sourceImg = cv2.imread("../Assignment 3 Pics/SourceImage.jpg")
sourceImg = cv2.cvtColor(sourceImg, cv2.COLOR_BGR2RGB)
targetImg = cv2.imread("../Assignment 3 Pics/TargetImage.jpg")
targetImg = cv2.cvtColor(targetImg, cv2.COLOR_BGR2RGB)


def calculateScore(source: np.ndarray, target: np.ndarray):
    score = source[:, 1] - target[:, 1]
    score = np.sum(np.abs(score))
    return score


if __name__ == "__main__":
    good = False
    bestScore = float("inf")

    sift = cv2.SIFT.create()
    graySource = cv2.cvtColor(sourceImg.copy(), cv2.COLOR_RGB2GRAY)
    grayTarget = cv2.cvtColor(targetImg.copy(), cv2.COLOR_RGB2GRAY)
    kpS, desS = sift.detectAndCompute(graySource, mask=None)
    kpT, desT = sift.detectAndCompute(grayTarget, mask=None)


    best = (None, None)
    rng = np.random.randint
    matches = None
    iteration = 0
    while not (bestScore < 20_000 or iteration > 1_000_000):
        sourceIndex = rng(low=0, high=desS.shape[0], size=100)
        targetIndex = rng(low=0, high=desT.shape[0], size=100)
        descriptorSourceCopy = desS[sourceIndex, :]
        descriptorTargetCopy = desT[targetIndex, :]
        bf = cv2.BFMatcher.create(cv2.NORM_L2)
        matches = bf.match(descriptorSourceCopy, descriptorTargetCopy)
        matches = np.array(matches)
        ptS = list()
        ptT = list()

        currentKP = list()
        for match in matches:
            realSourceIndex = sourceIndex[match.queryIdx]
            realTargetIndex = targetIndex[match.trainIdx]
            currentKP.append((kpS[realSourceIndex], kpT[realTargetIndex]))
            ptS.append(kpS[realSourceIndex].pt)
            ptT.append(kpT[realTargetIndex].pt)

        ptS = np.array(ptS)
        ptT = np.array(ptT)
        score = calculateScore(ptS, ptT)
        if score < bestScore:
            best = currentKP
            bestScore = score
            print(bestScore)
        iteration += 1

    best = np.array(best)
    img_matches = cv2.drawMatches(sourceImg, best[:, 0], targetImg, best[:, 1], matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS |
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    main.manageImage(["RANSAC Matches"], [img_matches], display_result=False,
                     save_result=True, file_title="RANSAC Matches", compute=True, dpi=600)
