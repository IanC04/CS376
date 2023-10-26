import cv2
import numpy as np
import matplotlib.pyplot as plt

defaultImg = cv2.imread("../Assignment 3 Pics/Calibration.jpg")
defaultImg = cv2.cvtColor(defaultImg, cv2.COLOR_BGR2RGB)
window_name = "keypoints_window"

def get2D() -> np.ndarray:
    pass


def get3D() -> np.ndarray:
    pass

def displayImage(plot, index, img: np.ndarray, title: str) -> None:
    if img is None:
        raise ValueError("Image is None in displayImage()")

    plot[index].set_title(title)
    plot[index].imshow(img)

    # # Waits for keypress to prevent instant closing
    # cv2.waitKey(0)
    # # Close all open windows
    # cv2.destroyAllWindows()


def calculate(img: np.ndarray = defaultImg) -> (np.ndarray, np.ndarray):
    # Two plots for the images
    fx, plots = plt.subplots(1, 2, figsize=(20, 10))

    # Convert to grayscale image
    sift = cv2.SIFT.create()

    # CV2 imread stores in BGR format, but we changed to RGB, so we need to convert to grayscale
    grayImg = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp, des = sift.detectAndCompute(grayImg, None)

    kp = np.array(kp)
    des = np.array(des)

    img = cv2.drawKeypoints(grayImg, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    displayImage(plots, 0, img, "All Keypoints")

    kp_res = np.array([k.response for k in kp])
    sorted_indices = kp_res.argsort()

    kp_threshold = kp[sorted_indices[-100:]]
    des_threshold = des[sorted_indices[-100:]]

    img_threshold = cv2.drawKeypoints(grayImg, kp_threshold, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    displayImage(plots, 1, img_threshold, "Top 100 Keypoints")

    two_d = get2D()

    three_d = get3D()

    # Keypoints figure
    plt.savefig("../Output Pictures/keypoints.png", dpi=1200)
    return two_d, three_d

if __name__ == "__main__":
    calculate()
