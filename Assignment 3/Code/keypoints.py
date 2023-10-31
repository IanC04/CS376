import cv2
import numpy as np
import matplotlib.pyplot as plt

calibrationImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
calibrationImg = cv2.cvtColor(calibrationImg, cv2.COLOR_BGR2RGB)

# Center of the Calibration checkerboard box
center = (1971, 1865)

NEON_GREEN = (57, 255, 0)


def get2D(kp: np.ndarray) -> np.ndarray:
    '''
    Get the coordinates of the keypoints rounded to the nearest pixel
    :param kp: Keypoints which contain the (x, y) coordinates to round
    :return:
    '''
    kp_coords = np.array([k.pt for k in kp])
    kp_coords = np.array([(round(x) - center[0], round(y) - center[1]) for x, y in kp_coords]).T
    return kp_coords


def get3D(kp_px: np.ndarray) -> np.ndarray:
    pass


def calculate(img: np.ndarray = calibrationImg) -> (np.ndarray, np.ndarray):
    # Convert to grayscale image
    sift = cv2.SIFT.create()

    # CV2 imread stores in BGR format, but we changed to RGB, so we need to convert to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(grayImg, None)

    kp = np.array(kp)
    des = np.array(des)
    kp_img = cv2.drawKeypoints(img, kp, None, color=NEON_GREEN, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    kp_res = np.array([k.response for k in kp])
    sorted_kp_indices = kp_res.argsort()

    kp_threshold = kp[sorted_kp_indices[-20:]]
    des_threshold = des[sorted_kp_indices[-20:]]
    kp_img_threshold = cv2.drawKeypoints(img, kp_threshold, None, color=NEON_GREEN,
                                         flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the images
    displayImage(["All Keypoints", "Top 20 Keypoints"], kp_img, kp_img_threshold, display_result=False,
                 save_result=True, file_title="keypoints")

    two_d = get2D(kp_threshold)
    three_d = get3D(kp_threshold)
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
