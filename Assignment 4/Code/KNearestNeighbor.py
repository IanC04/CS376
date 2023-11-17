import numpy as np
import LoadImages
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm


def euclidean_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    A Hyperparameter
    Returns the Euclidean distance between two images, square root of each pixel's RGB summed up
    :param img1:
    :param img2:
    :return:
    """
    # Slower for some reason: return np.linalg.norm(img1 - img2, axis=1, ord=norm_type)
    return np.sqrt(np.sum(np.square(img1 - img2), axis=1))


def manhattan_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    A Hyperparameter
    Returns the Manhattan distance between two images, sum of each pixel's RGB
    :param img1:
    :param img2:
    :return:
    """
    return np.sum(np.abs(img1 - img2), axis=1)


def classify(training_data: np.ndarray, img_data: np.ndarray, K) -> np.ndarray:
    # top_matches.clear()
    img = np.expand_dims(img_data, axis=0)
    img = pca.transform(img)
    distances = distance(training_data, img)
    return np.argpartition(distances, K)[:K]


def calculate(training_data, training_labels, testing_data, testing_labels, K, print_results=False):
    correct = 0
    incorrect = 0
    accuracies = np.zeros(len(testing_data))
    predicted = np.zeros(len(testing_data), dtype=np.int8)
    for index in tqdm(range(len(testing_data)), desc="Testing K Nearest Neighbor"):
        closestK = classify(training_data, testing_data[index], K)
        est = training_labels[closestK].max()
        if est == testing_labels[index]:
            correct += 1
        else:
            incorrect += 1
        accuracy = correct / (correct + incorrect)
        predicted[index] = est
        if print_results:
            print(f"Incorrect: {incorrect}, Correct: {correct}, Accuracy: {accuracy}")
        accuracies[index] = accuracy

    return accuracies, predicted


def computeKNN(K, current_norm, training_data, training_labels, testing_data, testing_labels, labels):
    global norm
    norm = "L1" if current_norm == 1 else "L2"
    global distance
    distance = manhattan_distance if norm == "L1" else euclidean_distance
    print(f"KNearestNeighbor: {K}")

    training_data = pca.fit_transform(training_data)
    accuracy, predictions = calculate(training_data, training_labels, testing_data, testing_labels, K)

    return accuracy, predictions


def displayResults(name, K, accuracy, display: bool=False):
    if display == False:
        return
    figure, axis = plt.subplots(1, 4, squeeze=False)
    axis[0, 0].set_xlabel("Iterations")
    axis[0, 0].set_ylabel("Accuracy")
    iteration = 0
    print(f"Norm: {norm}")
    for i in range(3, 11, 2):
        K = i
        print(f"K: {K}")
        accuracy = np.insert(accuracy, 0, 0)
        axis[0, iteration].plot(accuracy)
        axis[0, iteration].set_xlim([1, len(accuracy) - 1])
        axis[0, iteration].set_ylim([0, 1])
        axis[0, iteration].set_title(f"K: {K} | {norm}")
        iteration += 1
    plt.savefig(f"../Output Pictures/Nearest Neighbors {norm}_{name}.png")
    # plt.show()
    plt.close()


# top_matches = MaxHeap.MaxHeap(K)
norm = "L1"  # L1 or L2
distance = manhattan_distance if norm == "L1" else euclidean_distance

pca = PCA(n_components=20)
# Uses pixel values to classify images
if __name__ == "__main__":
    # A Hyperparameter
    K = 7
    tr_d, tr_l, te_d, te_l, all_labels = LoadImages.all_images()
    computeKNN(K, 1, tr_d, tr_l, te_d, te_l, all_labels)
