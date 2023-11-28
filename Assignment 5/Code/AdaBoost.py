"""
Written by Ian Chen on 11/25/2023
GitHub: https://github.com/IanC04
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import LoadImages


class AdaBoost:
    """
    AdaBoost class
    """

    def __init__(self, training_folds, training_images, testing_folds, testing_images,
                 window_size=(24, 24)):
        """
        Initialize the AdaBoost class
        :param folds: array of folds in the form: (x1, y1, x2, y2)
        :param images:
        :param window_size:
        """
        self.original_training_folds = training_folds
        self.original_training_images = training_images
        self.original_testing_folds = testing_folds
        self.original_testing_images = testing_images
        self.window_size = window_size
        self.haar_indices = self.generate_haar_indicies(self.window_size)
        """
        Requires modification of original data to have non-face images
        Inspired by https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi
        =d0ea54f1a97a1922c679f88c6167328da472a259
        """
        self.training_faces, self.training_non_faces = self.generate_gray_and_scaled_faces(
            self.original_training_folds, self.original_training_images, self.window_size)
        self.testing_faces, self.testing_non_faces = self.generate_gray_and_scaled_faces(
            self.original_testing_folds, self.original_testing_images, self.window_size)

        self.training_face_iimages = self.generate_integral_images("training_faces",
                                                                   self.training_faces)
        self.training_non_face_iimages = self.generate_integral_images("training_non_faces",
                                                                       self.training_non_faces)
        self.testing_face_iimages = self.generate_integral_images("testing_faces",
                                                                  self.testing_faces)
        self.testing_non_face_iimages = self.generate_integral_images("testing_non_faces",
                                                                      self.testing_non_faces)
        pass

    @classmethod
    def generate_gray_and_scaled_faces(cls, folds, images, window_size: tuple) -> (
    np.ndarray, np.ndarray):
        """
        Generate faces from the folds and images
        :param folds:
        :param images:
        :return:
        """
        all_faces = list()
        all_non_faces = list()
        for index in tqdm(range(len(folds)), desc="Generating faces"):
            for img_name in images[index].keys():
                img = cls.rgb2gray(images[index][img_name])
                faces, non_faces = cls.get_faces(folds[index][img_name], img, window_size)
                all_faces.extend(faces)
                all_non_faces.extend(non_faces)

        all_faces = np.array(all_faces)
        all_non_faces = np.array(all_non_faces)
        return all_faces, all_non_faces

    @classmethod
    def get_faces(cls, folds, img, window_size: tuple):
        """
        Get the face from the fold and image
        :param fold:
        :param img:
        :return:
        """
        temp_img = img.copy()
        faces, non_faces = list(), list()
        for i, fold in enumerate(folds):
            # Convert [x1, y1, x2, y2] to int([y1, x1, y2, x2])
            x1, y1, x2, y2 = int(fold[0]), int(fold[1]), int(fold[2]), int(fold[3])
            face = temp_img[y1:y2, x1:x2]
            faces.append(cv2.resize(face, window_size))

            # See if the face is correct
            plt.imshow(faces[i], cmap="gray")
            plt.waitforbuttonpress()

            temp_img[y1:y2, x1:x2] = 0
            non_face = temp_img
            non_faces.extend(cls.get_windows(non_face, window_size))
        return faces, non_faces

    @classmethod
    def get_windows(cls, img, window_size: tuple) -> list:
        """
        Get the non faces from the fold and image
        :param fold:
        :param img:
        :return:
        """
        non_faces = list()
        height, width = img.shape
        for x in range(0, width - window_size[0], window_size[0]):
            for y in range(0, height - window_size[1], window_size[1]):
                # TODO: Find how to generate non faces
                non_faces.append(cv2.resize(img[y:y + window_size[1], x:x + window_size[0]],
                                            window_size))
                plt.imshow(non_faces[-1], cmap="gray")
                plt.waitforbuttonpress()
        return non_faces

    def train(self):
        """
        Train the model
        """
        pass

    def get_best_features(self, amount):
        """
        Get the best features from the images
        Section 3 of AdaBoost paper employs an AdaBoosting algorithm
        to select the best features
        :param amount:
        :return:
        """
        pass

    def predict(self):
        """
        Predict the faces in the images
        """
        pass

    @classmethod
    def generate_integral_images(cls, file_name, imgs: np.ndarray) -> np.ndarray:
        """
        Generate integral images from images
        :param imgs:
        :return:
        """
        if os.path.exists(f"{LoadImages.CACHE_PATH}/{file_name}_integral_images.npy"):
            with open(f"{LoadImages.CACHE_PATH}/{file_name}_integral_images.npy", "rb") as f:
                print("Cache found. Loading integral images...")
                integral_images = np.load(f, allow_pickle=True)
        else:
            print("Cache not found. Generating integral images...")
            integral_images = cls.convert_to_integral(imgs)
            with open(f"{LoadImages.CACHE_PATH}/{file_name}_integral_images.npy", "wb") as f:
                np.save(f, integral_images)
        return integral_images

    @classmethod
    def convert_to_integral(cls, imgs: np.ndarray) -> np.ndarray:
        """
        Convert images to integral images
        :param imgs:
        :return:
        """
        integral_imgs = list()
        for index in tqdm(range(len(imgs)), desc="Converting to integral images"):
            i_imgs = dict()
            for img_name in imgs[index].keys():
                i_imgs[img_name] = cls.get_integral_image(imgs[index][img_name])
            integral_imgs.append(i_imgs)
        integral_imgs = np.array(integral_imgs)
        return integral_imgs

    @classmethod
    def rgb2gray(cls, img: np.ndarray) -> np.ndarray:
        """
        Convert RGB image to grayscale
        :param img:
        :return:
        """
        return np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])

    @classmethod
    def get_integral_image(cls, img: np.ndarray) -> np.ndarray:
        """
        Get the integral image from an image
        :param img:
        :return:
        """
        assert img.shape[2] == 3, f"Image shape is not 3: {img.shape}"
        img = cls.rgb2gray(img)
        integral_img = np.cumsum(np.cumsum(img, axis=0), axis=1)
        return integral_img

    @classmethod
    def generate_haar_indicies(cls, size: tuple):
        """
        Generate the haar indices
        :return:
        """
        two_piece = []
        three_piece = []
        four_piece = []
        if os.path.isfile(f"{LoadImages.CACHE_PATH}/haar_indices.npy"):
            with open(f"{LoadImages.CACHE_PATH}/haar_indices.npy", 'rb') as f:
                print("Cache found. Loading Haar Indices...")
                two_piece = np.load(f)
                three_piece = np.load(f)
                four_piece = np.load(f)
        else:
            width, height = size
            for x in tqdm(range(width), desc="Getting Haar Indices"):
                for y in range(height):
                    for w in range(1, width):
                        for h in range(1, height):
                            if y + h - 1 < height and x + 2 * w - 1 < width:
                                # Horizontal-2
                                two_piece.append(
                                    (y, x, y + h - 1, x + w - 1, y, x + w, y + h - 1,
                                     x + 2 * w - 1))
                            if y + h - 1 < height and x + 3 * w - 1 < width:
                                # Horizontal-3
                                three_piece.append(
                                    (y, x, y + h - 1, x + w - 1, y, x + w, y + h - 1,
                                     x + 2 * w - 1,
                                     y, x + 2 * w, y + h - 1, x + 3 * w - 1))
                            if y + 2 * h - 1 < height and x + w - 1 < width:
                                # Vertical-2
                                two_piece.append(
                                    (y, x, y + h - 1, x + w - 1, y + h, x, y + 2 * h - 1,
                                     x + w - 1))
                            if y + 3 * h - 1 < height and x + w - 1 < width:
                                # Vertical-3
                                three_piece.append(
                                    (y, x, y + h - 1, x + w - 1, y + h, x, y + 2 * h - 1,
                                     x + w - 1,
                                     y + 2 * h, x, y + 3 * h - 1, x + w - 1))
                            if y + 2 * h - 1 < height and x + 2 * w - 1 < width:
                                # Diagonal-4
                                four_piece.append(
                                    (y, x, y + h - 1, x + w - 1, y, x + w, y + h - 1,
                                     x + 2 * w - 1,
                                     y + h, x, y + 2 * h - 1, x + w - 1, y + h, x + w,
                                     y + 2 * h - 1, x + 2 * w - 1))
            two_piece = np.array(two_piece)
            three_piece = np.array(three_piece)
            four_piece = np.array(four_piece)
            with open(f"{LoadImages.CACHE_PATH}/haar_indices.npy", 'wb') as f:
                np.save(f, two_piece)
                np.save(f, three_piece)
                np.save(f, four_piece)
                print("Saved Haar Indices to Cache")
        return two_piece, three_piece, four_piece


if __name__ == "__main__":
    tr_folds, te_folds = LoadImages.load_folds()
    tr_images, te_images = LoadImages.load_images(np.concatenate((tr_folds, te_folds)))
    model = AdaBoost(tr_folds, tr_images, te_folds, te_images, window_size=(24, 24))
    model.train()
