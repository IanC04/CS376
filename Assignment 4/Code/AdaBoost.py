import LoadImages
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt


def get_haar_features(width, height) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Returns the Haar features of an image
    Separates the features into 2, 3, and 4 piece features
    Uses a subset of 2K per assignment guidelines
    Format is pairs of (y, x) = (Top Left Corner, Bottom Right Corner)
    :param width:
    :param height:
    :return:
    """
    two_piece = []
    three_piece = []
    four_piece = []
    if os.path.isfile(f"{cache_dir}/haar_features.npy"):
        with open(f"{cache_dir}/haar_features.npy", 'rb') as f:
            two_piece = np.load(f)
            three_piece = np.load(f)
            four_piece = np.load(f)
    else:
        for x in tqdm(range(width), desc="Getting Haar Features"):
            for y in range(height):
                for w in range(1, width - x):
                    for h in range(1, height - y):
                        if y + h - 1 < height and x + 2 * w - 1 < width:
                            # Horizontal-2
                            two_piece.append(
                                (y, x, y + h - 1, x + w - 1, y, x + w, y + h - 1, x + 2 * w - 1))
                            if y + h - 1 < height and x + 3 * w - 1 < width:
                                # Horizontal-3
                                three_piece.append(
                                    (y, x, y + h - 1, x + w - 1, y, x + w, y + h - 1, x + 2 * w - 1,
                                     y, x + 2 * w, y + h - 1, x + 3 * w - 1))
                        if y + 2 * h - 1 < height and x + w - 1 < width:
                            # Vertical-2
                            two_piece.append(
                                (y, x, y + h - 1, x + w - 1, y + h, x, y + 2 * h - 1, x + w - 1))
                            if y + 3 * h - 1 < height and x + w - 1 < width:
                                # Vertical-3
                                three_piece.append(
                                    (y, x, y + h - 1, x + w - 1, y + h, x, y + 2 * h - 1, x + w - 1,
                                     y + 2 * h, x, y + 3 * h - 1, x + w - 1))
                        if y + 2 * h - 1 < height and x + 2 * w - 1 < width:
                            # Diagonal-4
                            four_piece.append(
                                (y, x, y + h - 1, x + w - 1, y, x + w, y + h - 1, x + 2 * w - 1,
                                 y + h, x, y + 2 * h - 1, x + w - 1, y + h, x + w,
                                 y + 2 * h - 1, x + 2 * w - 1))
        two_piece = np.array(two_piece)
        three_piece = np.array(three_piece)
        four_piece = np.array(four_piece)
        with open(f"{cache_dir}/haar_features.npy", 'wb') as f:
            np.save(f, two_piece)
            np.save(f, three_piece)
            np.save(f, four_piece)
    # Use a random subset of 2K features
    generator = np.random.default_rng()
    partition = generator.integers(2000, size=2)
    partition.sort()
    amount_each = [partition[0], partition[1] - partition[0], 2000 - partition[1]]
    two_piece = generator.choice(two_piece, size=amount_each[0], replace=False)
    three_piece = generator.choice(three_piece, size=amount_each[1], replace=False)
    four_piece = generator.choice(four_piece, size=amount_each[2], replace=False)
    return (two_piece, three_piece, four_piece)


def get_integral_images(imgs: np.ndarray) -> np.ndarray:
    """
    Returns the integral images of a set of images
    Converts the images to grayscale first
    Then reshape them to 32x32, which is the CIFAR-10 image size
    Then computes the integral images
    :param imgs:
    :return:
    """
    imgs = rgb2gray(imgs)
    imgs = imgs.reshape((imgs.shape[0], 32, 32))
    integral_imgs = np.cumsum(np.cumsum(imgs, axis=1), axis=2)
    return integral_imgs


def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    """
    Under the assumption that the images are RGB and of the form [r:[0:1024), g:[1024:2048), b:[2048:3072)]
    Converts them to grayscale
    """
    r, g, b = rgb[:, :1024], rgb[:, 1024:2048], rgb[:, 2048:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def compute_haar_features(imgs: np.ndarray) -> list:
    """
    Computes the Haar features of the images
    :param imgs:
    :return:
    """
    integral_imgs = get_integral_images(imgs)
    haar_features = get_haar_features(32, 32)

    images_haar = []
    haar_feat = None
    for ii in tqdm(integral_imgs, desc="Computing Haar Features"):
        for coord_type in haar_features:
            for coord_set in coord_type:
                # 2 Rectangles
                if len(coord_set) == 8:
                    rect1 = area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                    rect2 = area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                    haar_feat = rect2 - rect1
                # 3 Rectangles
                elif len(coord_set) == 12:
                    rect1 = area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                    rect2 = area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                    rect3 = area(ii, coord_set[8], coord_set[9], coord_set[10], coord_set[11])
                    haar_feat = rect2 - rect1 - rect3
                # 4 Rectangles
                elif len(coord_set) == 16:
                    rect1 = area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                    rect2 = area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                    rect3 = area(ii, coord_set[8], coord_set[9], coord_set[10], coord_set[11])
                    rect4 = area(ii, coord_set[12], coord_set[13], coord_set[14], coord_set[15])
                    haar_feat = rect2 + rect4 - rect1 - rect3
                else:
                    AssertionError(f"Invalid Haar Feature: {coord_set}")
                images_haar.append(haar_feat)
    return haar_features


def area(img, y1, x1, y2, x2):
    return (img[y2][x2] + img[y1][x1]) - (img[y1][x2] + img[y2][x1])


def classify():
    pass


def train(training_data: np.ndarray, training_labels: np.ndarray, testing_data: np.ndarray, testing_labels: np.ndarray,
          labels: np.ndarray):
    compute_haar_features(training_data)
    pass


def test():
    pass


classifier = None
cache_dir = "../AdaBoostCache"
# Uses weak classifiers to classify images
if __name__ == "__main__":
    if not os.path.isdir(f"../AdaBoostCache"):
        os.mkdir(f"../AdaBoostCache")
    print("AdaBoost.py")
    training_data, training_labels, testing_data, testing_labels, labels = LoadImages.all_images()
    train(training_data, training_labels, testing_data, testing_labels, labels)
    classifier = np.zeros((labels, 2))
    pass
