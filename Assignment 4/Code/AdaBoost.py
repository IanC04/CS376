import LoadImages
import numpy as np
from tqdm import tqdm
import os
import multiprocessing as mp


def get_haar_indices(width, height) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Returns the Haar features of an image
    Separates the features into two, three, and four piece features
    Use a subset of 2K per assignment guidelines for training
    Format is pairs of (y, x) = (Top Left Corner, Bottom Right Corner)
    :param width:
    :param height:
    :return:
    """
    two_piece = []
    three_piece = []
    four_piece = []
    if os.path.isfile(f"{CACHE_DIR}/haar_indices.npy"):
        with open(f"{CACHE_DIR}/haar_indices.npy", 'rb') as f:
            two_piece = np.load(f)
            three_piece = np.load(f)
            four_piece = np.load(f)
            print("Loaded Haar Indices from Cache")
    else:
        for x in tqdm(range(width), desc="Getting Haar Indices"):
            for y in range(height):
                for w in range(1, width):
                    for h in range(1, height):
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
        with open(f"{CACHE_DIR}/haar_indices.npy", 'wb') as f:
            np.save(f, two_piece)
            np.save(f, three_piece)
            np.save(f, four_piece)
            print("Saved Haar Indices to Cache")
    # Use a random subset of 2K features
    if os.path.isfile(f"{CACHE_DIR}/haar_indices_subset.npy"):
        with open(f"{CACHE_DIR}/haar_indices_subset.npy", 'rb') as f:
            two_piece = np.load(f)
            three_piece = np.load(f)
            four_piece = np.load(f)
            print("Loaded Haar Indices Subset from Cache")
    else:
        generator = np.random.default_rng()
        partition = generator.integers(FEATURE_SUBSET, size=2)
        partition.sort()
        amount_each = [partition[0], partition[1] - partition[0], FEATURE_SUBSET - partition[1]]
        two_piece = generator.choice(two_piece, size=amount_each[0], replace=False)
        three_piece = generator.choice(three_piece, size=amount_each[1], replace=False)
        four_piece = generator.choice(four_piece, size=amount_each[2], replace=False)
        with open(f"{CACHE_DIR}/haar_indices_subset.npy", 'wb') as f:
            np.save(f, two_piece)
            np.save(f, three_piece)
            np.save(f, four_piece)
            print("Saved Haar Indices Subset to Cache")
    return two_piece, three_piece, four_piece


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


def compute_haar_features(imgs: np.ndarray, testing: bool = False) -> np.ndarray:
    """
    Computes the Haar features of the images
    :param imgs:
    :return:
    """
    if not testing and os.path.isfile(f"{CACHE_DIR}/haar_features.npy"):
        with open(f"{CACHE_DIR}/haar_features.npy", 'rb') as f:
            haar_features = np.load(f)
            print("Loaded Haar Features from Cache")
            return haar_features
    elif testing and os.path.isfile(f"{CACHE_DIR}/haar_features_test.npy"):
        with open(f"{CACHE_DIR}/haar_features_test.npy", 'rb') as f:
            haar_features = np.load(f)
            print("Loaded Haar Features Testing from Cache")
            return haar_features
    else:
        integral_imgs = get_integral_images(imgs)
        # Do I use the same subset for every image?
        haar_indices = get_haar_indices(32, 32)

        haar_features = np.zeros((len(integral_imgs), FEATURE_SUBSET), dtype=np.float64)
        feature = None
        img_index, haar_index = 0, 0
        for ii in tqdm(integral_imgs, desc=f"Computing Haar Features of {"Testing" if testing else "Training"} set"):
            haar_index = 0
            for coord_type in haar_indices:
                for coord_set in coord_type:
                    # 2 Rectangles
                    if len(coord_set) == 8:
                        rect1 = area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                        rect2 = area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                        feature = rect2 - rect1
                    # 3 Rectangles
                    elif len(coord_set) == 12:
                        rect1 = area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                        rect2 = area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                        rect3 = area(ii, coord_set[8], coord_set[9], coord_set[10], coord_set[11])
                        feature = rect2 - rect1 - rect3
                    # 4 Rectangles
                    elif len(coord_set) == 16:
                        rect1 = area(ii, coord_set[0], coord_set[1], coord_set[2], coord_set[3])
                        rect2 = area(ii, coord_set[4], coord_set[5], coord_set[6], coord_set[7])
                        rect3 = area(ii, coord_set[8], coord_set[9], coord_set[10], coord_set[11])
                        rect4 = area(ii, coord_set[12], coord_set[13], coord_set[14], coord_set[15])
                        feature = rect2 + rect4 - rect1 - rect3
                    else:
                        AssertionError(f"Invalid Haar Feature: {coord_set}")
                    haar_features[img_index, haar_index] = feature
                    haar_index += 1
            img_index += 1
        if testing:
            with open(f"{CACHE_DIR}/haar_features_test.npy", 'wb') as f:
                np.save(f, haar_features)
                print("Saved Haar Features Test to Cache")
        else:
            with open(f"{CACHE_DIR}/haar_features.npy", 'wb') as f:
                np.save(f, haar_features)
                print("Saved Haar Features to Cache")
        return haar_features


def area(img, y1, x1, y2, x2):
    """
    Returns the area of a rectangle with the given coordinates of the top left: (y1, x1) and bottom right: (y2, x2)
    :param img:
    :param y1:
    :param x1:
    :param y2:
    :param x2:
    :return:
    """
    return (img[y2, x2] + img[y1, x1]) - (img[y1, x2] + img[y2, x1])


class WeakClassifier:
    """
    Represents a weak classifier
    """

    def __init__(self, parity: np.int8 = -1, feature_index: np.int64 = np.iinfo(np.int64).max,
                 threshold: float = np.nan, error: float = np.inf, alpha: float = np.nan):
        # (result:1/0, parity:1/-1, feature_index, threshold, error, alpha(post-training)
        self.parity = parity
        self.feature_index = feature_index
        self.threshold = threshold
        self.error = error
        self.alpha = alpha

    def __str__(self):
        return (f"Parity: {self.parity}, Feature Index: {self.feature_index}, Threshold: {self.threshold}, "
                f"Error: {self.error}, Alpha: {self.alpha}")


def train_weak_classifier(weak_classifier: WeakClassifier, feature_per_image: np.ndarray, f_index,
                          binary_training_labels:
                          np.ndarray, weights: np.ndarray) -> WeakClassifier:
    """
    Trains a weak classifier
    :param feature_per_image: The feature to train on, includes all images
    :param f_index: The feature to train on, includes all images
    :param binary_training_labels:
    :param weights:
    :return:
    """

    all_thresholds = np.unique(feature_per_image)
    translation = np.min(all_thresholds)
    scale = np.max(all_thresholds - translation)

    scale_factor = 10
    for i in range(scale_factor):
        # Find the best threshold
        parity = 1
        threshold = (i / scale_factor) * scale + translation
        predictions = np.zeros(len(binary_training_labels))
        predictions[feature_per_image >= threshold] = 1
        error = np.sum(weights[binary_training_labels != predictions])
        if error > 0.5:
            error = 1 - error
            parity = -1
        if error < weak_classifier.error:
            weak_classifier.parity = parity
            weak_classifier.feature_index = f_index
            weak_classifier.threshold = threshold
            weak_classifier.error = error
    return weak_classifier


def train_binary_classifier(haar_features: np.ndarray, binary_training_labels: np.ndarray, T: np.uint16,
                            queue: mp.Queue = None, index: int = None) -> np.ndarray:
    """
    Trains a binary classifier
    l = number of positive examples
    m = number of negative examples
    :param haar_features:
    :param binary_training_labels:
    :param T: Number of weak classifiers and "time" steps
    :return:
    """
    l = np.count_nonzero(binary_training_labels)
    m = len(binary_training_labels) - l

    weights = np.array([1 / (2 * l) if label == 1 else 1 / (2 * m) for label in binary_training_labels])
    strong_classifier = []

    for _ in tqdm(range(T), desc=f"Training Binary Classifier: {index}"):
        # Normalize the weights
        weights /= np.sum(weights)

        weak_classifier = WeakClassifier()
        for feature_index in range(haar_features.shape[1]):
            single_feature = haar_features[:, feature_index]

            train_weak_classifier(weak_classifier, single_feature, feature_index,
                                  binary_training_labels, weights)
        beta = weak_classifier.error / (1 - weak_classifier.error)
        # weak_classifier.alpha = np.log((1 / beta))
        weak_classifier.alpha = -np.log(beta)

        strong_classifier.append(weak_classifier)
        predictions = np.zeros(len(binary_training_labels), dtype=np.uint8)
        predictions[weak_classifier.parity * haar_features[:, weak_classifier.feature_index] >=
                    weak_classifier.parity * weak_classifier.threshold] = 1
        print(weak_classifier.error)

        for i in range(len(weights)):
            if predictions[i] == binary_training_labels[i]:
                weights[i] *= beta
        # weights *= np.power(beta, predictions)

    strong_classifier = np.array(strong_classifier)
    queue.put((index, strong_classifier))
    return strong_classifier


def train(training_data: np.ndarray, training_labels: np.ndarray, labels: np.ndarray, number_of_weak_classifiers: int =
10):
    """
    Trains the AdaBoost classifier
    :param training_data:
    :param training_labels:
    :param labels:
    :return:
    """
    # Check if trained
    if TRAINED and os.path.isfile(f"{CACHE_DIR}/binary_classifiers.npy"):
        with open(f"{CACHE_DIR}/binary_classifiers.npy", 'rb') as f:
            all_strong_classifiers = np.load(f, allow_pickle=True)
            print("Loaded Binary Classifiers from Cache")
            return all_strong_classifiers
    else:
        T = np.uint16(number_of_weak_classifiers)
        assert T < FEATURE_SUBSET, f"T must be less than the feature subset size of {FEATURE_SUBSET}"
        haar_features = compute_haar_features(training_data)

        # Remove features that are all 0
        zero_indices = np.where(np.all(np.isclose(haar_features, 0), axis=0))
        haar_features = np.delete(haar_features, zero_indices, axis=1)

        # Rows of classifiers are strong classifiers
        all_strong_classifiers = np.ndarray((len(labels), T), dtype=WeakClassifier)

        queue = mp.Queue()
        processes = []
        for index, l in enumerate(labels):
            print(f"\nTraining {l}")
            binary_training_labels = np.where(training_labels == index, 1, 0)
            processes.append(mp.Process(target=train_binary_classifier, args=(haar_features, binary_training_labels,
                                                                              T, queue, index)))
            # strong_classifier = train_binary_classifier(haar_features, binary_training_labels, T)
            # all_strong_classifiers[index, :] = strong_classifier

        in_progress = len(processes)
        for p in processes:
            print(f"Starting Process {in_progress}")
            p.start()
            in_progress -= 1
        results = [queue.get() for p in processes]
        for p in processes:
            p.join()
            print(f"Process {in_progress} Complete")
            in_progress += 1
        results.sort(key=lambda x: x[0])
        results = [result[1] for result in results]
        for index, result in enumerate(results):
            all_strong_classifiers[index, :] = result
        with open(f"{CACHE_DIR}/binary_classifiers.npy", 'wb') as f:
            np.save(f, all_strong_classifiers)
            print("Saved Binary Classifiers to Cache")
    return all_strong_classifiers


def classify_image(features: np.ndarray, binary_classifiers: np.ndarray, correct_label) -> np.int8:
    binary_predictions = np.zeros(len(binary_classifiers))
    scores = np.zeros(len(binary_classifiers))
    sum_alpha_h, sum_alpha = 0, 0
    for index, strong_classifier in enumerate(binary_classifiers):
        sum_alpha_h = 0
        sum_alpha = 0
        for weak_classifier in strong_classifier:
            if features[weak_classifier.feature_index] >= weak_classifier.threshold:
                sum_alpha_h += weak_classifier.alpha
            sum_alpha += weak_classifier.alpha
        if sum_alpha_h > .5 * sum_alpha:
            binary_predictions[index] = 1
            scores[index] = sum_alpha_h / sum_alpha
    assert sum_alpha_h is not np.nan, "Sum Alpha H is NaN"
    assert sum_alpha_h is not np.inf, "Sum Alpha H is Inf"
    assert sum_alpha is not np.nan, "Sum Alpha is NaN"
    assert sum_alpha is not np.inf, "Sum Alpha is Inf"
    prediction = np.argmax(scores)
    # print(f"Alpha_H: {sum_alpha_h} | Alpha: {sum_alpha}")
    return prediction


def test(testing_data: np.ndarray, testing_labels: np.ndarray):
    if os.path.isfile(f"{CACHE_DIR}/binary_classifiers.npy"):
        with open(f"{CACHE_DIR}/binary_classifiers.npy", 'rb') as f:
            all_classifiers = np.load(f, allow_pickle=True)
            print("Loaded Binary Classifiers from Cache")
    else:
        raise FileNotFoundError("Binary Classifiers not found in cache")
    haar_features = compute_haar_features(testing_data, testing=True)
    predictions = np.zeros(len(testing_labels), dtype=np.uint8)
    for img in range(testing_data.shape[0]):
        predictions[img] = classify_image(haar_features[img, :], all_classifiers, testing_labels[img])
    accuracy = np.sum(predictions == testing_labels) / len(testing_labels)
    print(f"Accuracy: {accuracy}")
    return accuracy, predictions, testing_labels


def train_and_test(t, training_data: np.ndarray, training_labels: np.ndarray, testing_data: np.ndarray,
                   testing_labels: np.ndarray, label_names: np.ndarray):
    global TRAINED
    TRAINED = False
    train(training_data, training_labels, label_names, t)
    accuracy, predictions, testing_labels = test(testing_data, testing_labels)
    TRAINED = True
    return accuracy


CACHE_DIR = "../AdaBoostCache"
FEATURE_SUBSET = 2000
TRAINED = False

# Uses weak classifiers to classify images
if __name__ == "__main__":
    if not os.path.isdir(f"{CACHE_DIR}"):
        os.mkdir(f"{CACHE_DIR}")
    print("AdaBoost.py")
    tr_d, tr_l, te_d, te_l, label_names = LoadImages.all_images()
    if not TRAINED:
        binary_classifiers = train(tr_d, tr_l, label_names)
    test(te_d, te_l)

    import CrossValidation

    CrossValidation.confusion_matrix_adaboost(tr_d, tr_l, te_d, te_l, label_names)
    del CrossValidation
