import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from skimage.feature import haar_like_feature
from skimage.transform import integral_image

import numpy as np
import cv2
from tqdm import tqdm
from matplotlib import pyplot as plt

if __name__ == "__main__":
    from AdaBoost import AdaBoost
    from AdaBoost import DecisionStump

    custom_model = AdaBoost.load_model()
    del DecisionStump

    original_training_imgs = list()
    for layer in custom_model.original_training_images:
        for img in layer:
            original_training_imgs.append(layer[img])

    original_testing_imgs = list()
    for layer in custom_model.original_testing_images:
        for img in layer:
            original_testing_imgs.append(layer[img])
    # training_imgs = np.concatenate((custom_model.training_faces, custom_model.training_non_faces))
    # testing_imgs = np.concatenate((custom_model.testing_faces, custom_model.testing_non_faces))

    haar_cascade = cv2.CascadeClassifier('../Assignment 5 Pics/haarcascade_frontalface_default.xml')

    for img in original_testing_imgs:
        gray_img = AdaBoost.rgb2gray(img).astype(np.uint8)
        faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
        for (x, y, w, h) in faces_rect:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

        # plt.imshow(img)
        # plt.waitforbuttonpress()
