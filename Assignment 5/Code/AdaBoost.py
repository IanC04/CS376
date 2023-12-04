"""
Written by Ian Chen on 11/25/2023
GitHub: https://github.com/IanC04
"""
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import timeit

import LoadImages


def load():
    """
    Load the model
    """
    import pickle
    with open(f"{LoadImages.CACHE_PATH}/model.pkl", "rb") as f:
        print("Loading model...")
        m = pickle.load(f)
    del pickle
    return m


class AdaBoost:
    """
    AdaBoost class
    """

    def __init__(self, training_folds, training_images, testing_folds, testing_images,
                 window_size=(24, 24)):
        """
        Initialize the AdaBoost class
        :param training_folds: array of folds in the form: (x1, y1, x2, y2)
        :param testing_folds: array of folds in the form: (x1, y1, x2, y2)
        :param training_images:
        :param testing_images:
        :param window_size:
        """
        self.original_training_folds = training_folds
        self.original_training_images = training_images
        self.original_testing_folds = testing_folds
        self.original_testing_images = testing_images
        self.window_size = window_size

        # Stored with the format: (y1, x1, y2, x2)
        self.haar_indices = self.generate_haar_indicies(self.window_size)
        self.haar_indices_subset = self.generate_haar_indices_subset(self.haar_indices)

        """
        Requires modification of original data to have non-face images, windows where no face is 
        shown
        Inspired by https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi
        =d0ea54f1a97a1922c679f88c6167328da472a259
        """
        self.training_faces, self.training_non_faces = self.generate_gray_and_scaled_faces(
            "training", self.original_training_folds, self.original_training_images,
            self.window_size)
        self.testing_faces, self.testing_non_faces = self.generate_gray_and_scaled_faces(
            "testing", self.original_testing_folds, self.original_testing_images, self.window_size)

        self.training_face_iimages, self.training_non_face_iimages = self.generate_integral_images(
            "training",
            self.training_faces, self.training_non_faces)
        self.testing_face_iimages, self.testing_non_face_iimages = self.generate_integral_images(
            "testing",
            self.testing_faces, self.testing_non_faces)

        self.training_face_haar_features, self.training_non_face_haar_features = (
            self.get_haar_features("training", self.haar_indices_subset, self.training_face_iimages,
                                   self.training_non_face_iimages))

        self.testing_face_haar_features, self.testing_non_face_haar_features = (
            self.get_haar_features("testing", self.haar_indices_subset, self.testing_face_iimages,
                                   self.testing_non_face_iimages))
        self.strong_classifier = None

    @classmethod
    def generate_gray_and_scaled_faces(cls, file_name, folds, images, window_size: tuple) -> (
            np.ndarray, np.ndarray):
        """
        Generate faces from the folds and images
        :param file_name:
        :param folds:
        :param images:
        :param window_size:
        :return:
        """
        if os.path.exists(f"{LoadImages.CACHE_PATH}/{file_name}_faces_and_non_faces.npy"):
            with open(f"{LoadImages.CACHE_PATH}/{file_name}_faces_and_non_faces.npy", "rb") as f:
                print(f"Cache found. Loading {file_name} faces...")
                all_faces = np.load(f, allow_pickle=True)
                print(f"Cache found. Loading {file_name} non faces...")
                all_non_faces = np.load(f, allow_pickle=True)
            return all_faces, all_non_faces
        else:
            print(f"Cache not found. Generating {file_name} faces...")
            all_faces = list()
            all_non_faces = list()
            for index in tqdm(range(len(folds)), desc="Generating faces"):
                for img_name in images[index].keys():
                    img = cls.rgb2gray(images[index][img_name])
                    print(img_name)
                    faces, non_faces = cls.get_faces_and_non_faces(folds[index][img_name], img,
                                                                   window_size)
                    all_faces.extend(faces)
                    all_non_faces.extend(non_faces)

            all_faces = np.array(all_faces)
            all_non_faces = np.array(all_non_faces)
            with open(f"{LoadImages.CACHE_PATH}/{file_name}_faces_and_non_faces.npy", "wb") as f:
                np.save(f, all_faces)
                np.save(f, all_non_faces)
            return all_faces, all_non_faces

    @classmethod
    def get_faces_and_non_faces(cls, folds, img, window_size: tuple):
        """
        Get the face from the fold and image
        Potentially normalize the results to have unit variance
        :param folds:
        :param img:
        :param window_size:
        :return:
        """
        temp_img = img.copy()
        faces, non_faces = list(), list()
        face_coord = None

        # print(f"Number of faces: {len(folds)}")
        for i, fold in enumerate(folds):
            # Convert [x1, y1, x2, y2] to int([x1, y1, x2, y2])
            x1, y1, x2, y2 = int(fold[0]), int(fold[1]), int(fold[2]), int(fold[3])
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, img.shape[1]), min(y2, img.shape[0])

            # Convert to row:column [y1, x1, y2, x2]
            face = temp_img[y1:y2, x1:x2]
            faces.append(cv2.resize(face, window_size))

            # Face coordinates
            face_coord = np.array([x1, y1, x2, y2])

            # See if the face is correct
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
            # plt.imshow(img, cmap="gray")
            # plt.waitforbuttonpress()
            # plt.imshow(faces[i], cmap="gray")
            # plt.waitforbuttonpress()

        if len(folds) < 2:
            non_face = temp_img
            non_faces.extend(cls.get_non_faces(non_face, window_size, face_coord))
        return faces, non_faces

    @classmethod
    def get_non_faces(cls, img: np.ndarray, window_size: tuple, face: np.ndarray) -> list:
        """
        Get the non faces from the image, get windows where minimal face is shown
        :param img:
        :param window_size:
        :param face: [X1, y1, x2, y2]
        :return:
        """
        # Check if face blacked out
        non_faces = list()

        # Flipped to scale down
        scale_x, scale_y = window_size[0] * 2 / img.shape[1], window_size[1] * 2 / img.shape[0]
        img = cv2.resize(img, (window_size[0] * 2, window_size[1] * 2))

        # Rounding issue?
        f_x1, f_y1, f_x2, f_y2 = np.round(face * [scale_x, scale_y, scale_x, scale_y]).astype(int)

        # cv2.rectangle(img, (f_x1, f_y1), (f_x2, f_y2), (255, 0, 0), 1)
        # plt.imshow(img, cmap="gray")
        # plt.waitforbuttonpress()
        for x in range(0, 2):
            for y in range(0, 2):
                if face is not None:
                    xi1, yi1, xi2, yi2 = (max(x * window_size[0], f_x1), max(y * window_size[
                        1], f_y1), min(x * window_size[0] + window_size[0], f_x2),
                                          min(y * window_size[1] + window_size[1], f_y2))
                    width = xi2 - xi1
                    height = yi2 - yi1
                    face_percentage = width * height / (window_size[0] * window_size[1])
                    # print(f"Face Percentage: {face_percentage}, Width: {width}, Height: {height}")
                    if (width > 0 and height > 0) and face_percentage > 0.1:
                        continue

                # Non-face is sections where not face is shown
                non_face = img[y * window_size[1]:y * window_size[1] + window_size[1],
                           x * window_size[0]:x * window_size[0] + window_size[0]]
                non_faces.append(non_face)
                # plt.imshow(non_faces[-1], cmap="gray")
                # plt.waitforbuttonpress()
        return non_faces

    def train(self):
        """
        Train the model
        """
        if os.path.isfile(f"{LoadImages.CACHE_PATH}/strong_classifier.npy"):
            with open(f"{LoadImages.CACHE_PATH}/strong_classifier.npy", "rb") as f:
                print(f"Cache found. Loading Strong Classifier...")
                self.strong_classifier = np.load(f)
        else:
            # Initialize weights
            m = self.training_face_haar_features.shape[0]
            l = self.training_non_face_haar_features.shape[0]
            training_features = np.concatenate((self.training_face_haar_features,
                                                self.training_non_face_haar_features))
            training_labels = np.concatenate((np.ones(m), np.zeros(l)))

            weights = np.array([1 / (2 * m) for _ in range(m)] + [1 / (2 * l) for _ in range(l)])

            # Strong classifier
            strong_classifier = list()

            for _ in tqdm(range(200), desc="Training strong classifier", position=0):
                # Normalize weights
                weights /= np.sum(weights)

                # Get best feature
                best_weak_classifier = self.get_best_feature(training_labels, training_features,
                                                             weights)
                strong_classifier.append(best_weak_classifier)

            self.strong_classifier = np.array(strong_classifier)
            with open(f"{LoadImages.CACHE_PATH}/strong_classifier.npy", "wb") as f:
                np.save(f, self.strong_classifier)
                print(f"Saved Strong Classifier to Cache")

    def get_best_feature(self, labels, all_features, weights):
        """
        Get the best feature from the images
        Section 3 of AdaBoost paper employs an AdaBoosting algorithm
        to select the best features
        :param labels:
        :param all_features:
        :param weights:
        :return:
        """
        weak_classifier = DecisionStump()
        # Get the best feature
        for idx in tqdm(range(len(all_features[0])), desc="Getting best feature", position=1,
                        leave=False):
            feature = all_features[:, idx].flatten()
            self.train_weak_classifier(weak_classifier, labels, feature, idx, weights)

        beta = weak_classifier.error / (1 - weak_classifier.error)
        assert 0 < beta < 1, f"Beta is {beta} for feature: {weak_classifier.feature_index}"
        weak_classifier.alpha = -np.log(1 / beta)
        assert weak_classifier.valid(), f"Weak Classifier is not valid: {weak_classifier}"

        # Update weights
        predictions = np.zeros(len(all_features))
        predictions[weak_classifier.parity * all_features[:, weak_classifier.feature_index] <
                    weak_classifier.parity * weak_classifier.threshold] = 1

        weights[predictions == labels] *= beta
        assert weights.all() >= 0, f"Weights are not greater than 0: {weights}"

        return weak_classifier

    def train_weak_classifier(self, classifier, labels, feature, feature_index, weights):
        """
        Train the weak classifier
        :param classifier:
        :param labels:
        :param feature:
        :param feature_index:
        :param weights:
        :return:
        """
        # Get the best threshold
        thresholds = np.unique(feature)
        min_threshold, max_threshold = np.min(thresholds), np.max(thresholds)

        # start = timeit.timeit()
        N = 100
        for percent in range(N):
            parity = 1
            threshold = min_threshold + percent * (max_threshold - min_threshold) / N
            predictions = np.zeros(len(feature))
            predictions[feature < threshold] = 1
            error = np.sum(weights[predictions != labels])

            if error > 0.5:
                error = 1 - error
                parity = -1
            assert 0 < error <= 0.5, (
                f"Error is not in [0, 0.5]: {error} for feature: {feature_index}")
            if error < classifier.error:
                classifier.parity = parity
                classifier.feature_index = feature_index
                classifier.threshold = threshold
                classifier.error = error

        # end = timeit.timeit()
        # print(end - start)

    def test(self):
        """
        Test the model
        """
        self.predict()
        pass

    def predict(self):
        """
        Predict the faces in the images
        """
        pass

    def save(self):
        """
        Save the model
        """
        import pickle
        with open(f"{LoadImages.CACHE_PATH}/model.pkl", "wb") as f:
            print("Saving model...")
            pickle.dump(self, f)
        del pickle

    @classmethod
    def generate_integral_images(cls, file_name, face_imgs: np.ndarray,
                                 non_face_imgs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate integral images from images
        :param file_name:
        :param face_imgs:
        :param non_face_imgs:
        :return:
        """
        if os.path.exists(f"{LoadImages.CACHE_PATH}/{file_name}_integral_images.npy"):
            with open(f"{LoadImages.CACHE_PATH}/{file_name}_integral_images.npy", "rb") as f:
                print(f"Cache found. Loading {file_name} integral images...")
                face_integral_images = np.load(f, allow_pickle=True)
                non_face_integral_images = np.load(f, allow_pickle=True)
                return face_integral_images, non_face_integral_images
        else:
            print(f"Cache not found. Generating {file_name} integral images...")
            face_integral_images = cls.convert_to_integral(face_imgs)
            non_face_integral_images = cls.convert_to_integral(non_face_imgs)

            with open(f"{LoadImages.CACHE_PATH}/{file_name}_integral_images.npy", "wb") as f:
                np.save(f, face_integral_images)
                np.save(f, non_face_integral_images)
        return face_integral_images, non_face_integral_images

    @classmethod
    def convert_to_integral(cls, imgs: np.ndarray) -> np.ndarray:
        """
        Convert images to integral images
        :param imgs:
        :return:
        """
        integral_imgs = np.zeros(imgs.shape)
        for index in tqdm(range(len(imgs)), desc="Converting to integral images"):
            i_imgs = cls.get_integral_image(imgs[index])
            integral_imgs[index] = i_imgs
        # integral_imgs = np.array(integral_imgs)
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
        Get the integral image from a grayscale image
        :param img: GRAYSCALE image that should already be scaled to the window size
        :return:
        """
        assert img.ndim == 2 and img.shape[0] == img.shape[1], f"Image shape is not 2: {img.shape}"
        # img = cls.rgb2gray(img)
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
        if os.path.isfile(f"{LoadImages.CACHE_PATH}/all_haar_indices.npy"):
            with open(f"{LoadImages.CACHE_PATH}/all_haar_indices.npy", 'rb') as f:
                print(f"Cache found. Loading Haar Indices...")
                two_piece = np.load(f)
                three_piece = np.load(f)
                four_piece = np.load(f)
        else:
            width, height = size
            for x in tqdm(range(width), desc="Getting Haar Indices"):
                for y in range(height):
                    for w in range(1, width + 1):
                        for h in range(1, height + 1):
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
            with open(f"{LoadImages.CACHE_PATH}/all_haar_indices.npy", 'wb') as f:
                np.save(f, two_piece)
                np.save(f, three_piece)
                np.save(f, four_piece)
                print(f"Saved Haar Indices to Cache")
        return two_piece, three_piece, four_piece

    @classmethod
    def generate_haar_indices_subset(cls, haar_indices: tuple[np.ndarray, np.ndarray, np.ndarray],
                                     subset_size=5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a subset of the haar indices, ideally with substantial rectangle sizes
        :param haar_indices:
        :param subset_size:
        :return:
        """
        if os.path.isfile(f"{LoadImages.CACHE_PATH}/haar_indices_subset.npy"):
            with open(f"{LoadImages.CACHE_PATH}/haar_indices_subset.npy", 'rb') as f:
                print(f"Cache found. Loading Haar Indices Subset...")
                two_piece_subset = np.load(f)
                three_piece_subset = np.load(f)
                four_piece_subset = np.load(f)
                return two_piece_subset, three_piece_subset, four_piece_subset
        else:
            print(f"Cache not found. Generating Haar Indices Subset...")
            two_piece, three_piece, four_piece = haar_indices

            generator = np.random.default_rng()
            partition = generator.integers(subset_size, size=2)
            partition.sort()
            two_piece_subset = generator.choice(two_piece, partition[0], replace=False)
            three_piece_subset = generator.choice(three_piece, partition[1] - partition[0],
                                                  replace=False)
            four_piece_subset = generator.choice(four_piece, subset_size - partition[1],
                                                 replace=False)
            with open(f"{LoadImages.CACHE_PATH}/haar_indices_subset.npy", 'wb') as f:
                np.save(f, two_piece_subset)
                np.save(f, three_piece_subset)
                np.save(f, four_piece_subset)
                print(f"Saved Haar Indices Subset to Cache")
            return two_piece_subset, three_piece_subset, four_piece_subset

    @classmethod
    def display_haar_rectangles(cls, img, features):
        """
        Display the haar rectangles, currently only supports two-piece haar features
        :param img:
        :param features:
        :return:
        """
        for i in features:
            random_image = img.copy()
            cv2.rectangle(random_image, (i[1], i[0]), (i[3], i[2]), (255, 0, 0), 1)
            cv2.rectangle(random_image, (i[5], i[4]), (i[7], i[6]), (0, 255, 0), 1)
            plt.imshow(random_image, cmap="gray")
            plt.waitforbuttonpress()

    @classmethod
    def get_haar_features(cls, file_name, haar_indices_subset, face_iimages, non_face_iimages) -> \
            tuple[np.ndarray, np.ndarray]:
        """
        Get the haar features from the images
        :param file_name:
        :param haar_indices_subset:
        :param face_iimages:
        :param non_face_iimages:
        :return:
        """

        if os.path.exists(f"{LoadImages.CACHE_PATH}/{file_name}_haar_features.npy"):
            with open(f"{LoadImages.CACHE_PATH}/{file_name}_haar_features.npy", "rb") as f:
                print(f"Cache found. Loading {file_name} Haar Features...")
                face_haar_features = np.load(f, allow_pickle=True)
                non_face_haar_features = np.load(f, allow_pickle=True)
                return face_haar_features, non_face_haar_features
        else:
            print(f"Cache not found. Generating {file_name} Haar Features...")

            face_haar_features = cls.get_haar_features_helper(haar_indices_subset, face_iimages)
            non_face_haar_features = cls.get_haar_features_helper(haar_indices_subset,
                                                                  non_face_iimages)
            assert face_haar_features is not None and non_face_haar_features is not None, \
                "Haar Features not generated"
            with open(f"{LoadImages.CACHE_PATH}/{file_name}_haar_features.npy", "wb") as f:
                np.save(f, face_haar_features)
                np.save(f, non_face_haar_features)
                print(f"Saved {file_name} Haar Features to Cache")
            return face_haar_features, non_face_haar_features

    @classmethod
    def get_haar_features_helper(cls,
                                 haar_indices_subset: tuple[np.ndarray, np.ndarray, np.ndarray],
                                 iimages: np.ndarray) -> np.ndarray:
        """
        Helper function for get_haar_features
        :param haar_indices_subset:
        :param iimages:
        :return:
        """
        haar_features = np.zeros((len(iimages), sum([arr.shape[0] for arr in haar_indices_subset])))

        feature = None
        for img_idx, ii in tqdm(enumerate(iimages), desc=f"Computing Haar Features"):
            coord_idx = 0
            for coord_type in haar_indices_subset:
                for coord_set in coord_type:
                    # 2 Rectangles
                    if len(coord_set) == 8:
                        rect1 = cls.area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                        rect2 = cls.area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                        feature = rect2 - rect1
                    # 3 Rectangles
                    elif len(coord_set) == 12:
                        rect1 = cls.area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                        rect2 = cls.area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                        rect3 = cls.area(ii, coord_set[8], coord_set[9], coord_set[10],
                                         coord_set[11])
                        feature = rect2 - rect1 - rect3
                    # 4 Rectangles
                    elif len(coord_set) == 16:
                        rect1 = cls.area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                        rect2 = cls.area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                        rect3 = cls.area(ii, coord_set[8], coord_set[9], coord_set[10],
                                         coord_set[11])
                        rect4 = cls.area(ii, coord_set[12], coord_set[13], coord_set[14],
                                         coord_set[15])
                        feature = rect2 + rect4 - rect1 - rect3
                    else:
                        AssertionError(f"Invalid Haar Feature: {coord_set}")
                    haar_features[img_idx, coord_idx] = feature

                    coord_idx += 1
        haar_features = np.array(haar_features)
        return haar_features

    @classmethod
    def area(cls, ii, y1, x1, y2, x2):
        """
        Get the area of the integral image
        :param ii:
        :param y1:
        :param x1:
        :param y2:
        :param x2:
        :return:
        """
        br = ii[y2, x2]
        tl = ii[y1 - 1, x1 - 1] if x1 > 0 and y1 > 0 else 0
        tr = ii[y1 - 1, x2] if y1 > 0 else 0
        bl = ii[y2, x1 - 1] if x1 > 0 else 0
        return (br + tl) - (tr + bl)


class DecisionStump:
    """
    Decision Stump class
    """

    def __init__(self):
        self.parity = None
        self.feature_index = None
        self.threshold = np.inf
        self.error = np.inf
        self.alpha = np.inf

    def valid(self):
        """
        Check if the decision stump is valid
        :return:
        """
        return (self.parity is not None and self.feature_index is not None and self.threshold !=
                np.inf and self.error != np.inf and self.alpha != np.inf)

    def __str__(self):
        """
        String representation of the Decision Stump
        :return:
        """
        return f"DecisionStump(parity={self.parity}, feature_index={self.feature_index}, " \
               f"threshold={self.threshold}, error={self.error}, alpha={self.alpha})"


if __name__ == "__main__":
    tr_folds, te_folds = LoadImages.load_folds()
    tr_images, te_images = LoadImages.load_images(np.concatenate((tr_folds, te_folds)))
    model = AdaBoost(tr_folds, tr_images, te_folds, te_images, window_size=(24, 24))
    model.train()
