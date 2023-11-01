import cv2
import numpy as np
import matplotlib.pyplot as plt

calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
calibrationImg = cv2.cvtColor(calibrationImg, cv2.COLOR_BGR2RGB)
NEON_GREEN = (57, 255, 0)


def get2D(img: np.ndarray) -> np.ndarray:
    """
    Get the coordinates of the keypoints in (x, y) starting from the top-left corner

    :param img: Image to get the keypoints from
    :return:
    """
    coords2D = np.array([[1971, 1377], [1441, 1558], [1706, 1559], [1971, 1555], [2236, 1553], [2499, 1553],
                         [1149, 1762], [2765, 1742], [1504, 2112], [1975, 2750]], dtype=np.int64).T
    return coords2D


def get3D(img: np.ndarray) -> np.ndarray:
    """
    Get the coordinates of the keypoints in (x, y, z) starting from the front middle of the calibration rig
    x points left, y points right, z points down
    :param img:
    :return:
    """
    coords3D = np.array([[6, 6, 0], [5, 1, 0], [4, 2, 0], [3, 3, 0], [2, 4, 0], [1, 5, 0],
                         [6, 0, 1], [0, 6, 1], [3, 0, 2], [0, 0, 4]], dtype=np.int64).T
    return coords3D


def calculate(img: np.ndarray = calibrationImg) -> (np.ndarray, np.ndarray):
    # # Convert to grayscale image
    # sift = cv2.SIFT.create()
    #
    # # CV2 imread stores in BGR format, but we changed to RGB, so we need to convert to grayscale
    # grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # kp, des = sift.detectAndCompute(grayImg, None)
    #
    # kp = np.array(kp)
    # des = np.array(des)
    # kp_img = cv2.drawKeypoints(img, kp, None, color=NEON_GREEN, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # kp_res = np.array([k.response for k in kp])
    # sorted_kp_indices = kp_res.argsort()
    #
    # kp_threshold = kp[sorted_kp_indices[-20:]]
    # des_threshold = des[sorted_kp_indices[-20:]]
    # kp_img_threshold = cv2.drawKeypoints(img, kp_threshold, None, color=NEON_GREEN,
    #                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #
    # # Display the images
    # displayImage(["All Keypoints", "Top 20 Keypoints"], kp_img, kp_img_threshold, display_result=False,
    #              save_result=True, file_title="keypoints")
    """
    Calculates the 2D and 3D coordinates of the keypoints in the image

    :param img:
    :return:
    """

    two_d = get2D(img)
    three_d = get3D(img)
    return two_d, three_d


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


if __name__ == "__main__":
    calculate()
