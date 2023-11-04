import cv2
import numpy as np
import matplotlib.pyplot as plt

window_name = "main_window"


def manageImage(titles: list, images: list, save_result: bool = False, display_result: bool = False,
                file_title: str = "Default", compute: bool = True) -> None:
    if not compute:
        return
    if images is None or len(images) == 0:
        raise ValueError("Invalid images in manageImage()")
    # Two plots for the images
    fx, plot = plt.subplots(1, len(images), figsize=(20, 10), squeeze=False)
    for index, img in enumerate(images):
        plot[0, index].set_title(titles[index])
        plot[0, index].imshow(img)

    if save_result:
        plt.savefig(f"../Output Pictures/{file_title}.png", dpi=1200)
    if display_result:
        plt.show()

def manageMatrix(titles: list, matrices: list, save_result: bool = False, display_result: bool = False,
                file_title: str = "Default", compute: bool = True) -> None:
    if not compute:
        return
    if matrices is None or len(matrices) == 0:
        raise ValueError("Invalid images in manageImage()")
    # Two plots for the images
    fx, plot = plt.subplots(1, len(matrices), figsize=(20, 10), squeeze=False)
    for index, matrix in enumerate(matrices):
        plot[0, index].set_title(titles[index])
        plot[0, index].matshow(matrix, cmap="Accent")
        for (i, j), z in np.ndenumerate(matrix):
            plot[0, index].text(j, i, '{:0.5f}'.format(z), ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    if save_result:
        plt.savefig(f"../Output Pictures/{file_title}.png", dpi=1200)
    if display_result:
        plt.show()
