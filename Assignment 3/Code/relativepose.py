import cv2
import numpy as np
import matplotlib.pyplot as plt

sourceImg = cv2.imread("../Assignment 3 Pics/SourceImage.jpg")
targetImg = cv2.imread("../Assignment 3 Pics/TargetImage.jpg")


def getRelativePose(sCoord2D: np.ndarray, tCoord2D: np.ndarray, K: np.ndarray) -> (np.ndarray, np.ndarray):
    pass


def calculate():
    pass

def displayImage(titles: list, *images: np.ndarray, save_result: bool = True, display_result: bool = False, file_title:
str = None) -> None:
    if images is None or len(images) == 0:
        raise ValueError("Invalid images in displayImage()")
    # Two plots for the images
    fx, plot = plt.subplots(1, len(images), figsize=(20, 10))
    for index, img in enumerate(images):
        plot[index].set_title(titles[index])
        plot[index].imshow(img)

    if save_result:
        # Keypoints figure
        plt.savefig(f"../Output Pictures/{file_title}.png", dpi=1200)
    if display_result:
        plt.show()