import cv2
import numpy as np
import matplotlib.pyplot as plt

window_name = "main_window"


def manageImage(titles: list, images: list, save_result: bool = False, display_result: bool = False,
                file_title: str = "Default") -> None:
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
