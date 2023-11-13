import numpy as np
import LoadImages
import matplotlib.pyplot as plt


def euclidean_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Returns the Euclidean distance between two images, square root of each pixel's RGB summed up
    Modified to include L1 Norm
    :param img1:
    :param img2:
    :return:
    """
    # L1 Norm: a = np.linalg.norm(img1.astype(np.int32) - img2.astype(np.int32), ord=1)
    # Slower: a = np.sqrt(np.sum(np.square(img1.astype(np.int32)-img2.astype(np.int32))))
    # Faster:  np.linalg.norm(img1.astype(np.int32) - img2.astype(np.int32))
    return np.linalg.norm(img1.astype(np.int32) - img2.astype(np.int32), axis=1, ord=norm_type)


def classify(training_data: np.ndarray, img_data: np.ndarray) -> np.ndarray:
    # top_matches.clear()
    img = np.expand_dims(img_data, axis=0)
    distances = euclidean_distance(training_data, img)
    return np.argpartition(distances, K)[:K]


def calculate(training_data, training_labels, testing_data, testing_labels):
    correct = 0
    incorrect = 0
    accuracies = np.zeros(len(testing_data))
    for index in range(len(testing_data)):
        closestK = classify(training_data, testing_data[index])
        est = training_labels[closestK].max()
        if est == testing_labels[index]:
            correct += 1
        else:
            incorrect += 1
        accuracy = correct / (correct + incorrect)
        print(f"Incorrect: {incorrect}, Correct: {correct}, Accuracy: {accuracy}")
        accuracies[index] = accuracy
    return accuracies


K = 7
# top_matches = MaxHeap.MaxHeap(K)
norm = "L2"  # L1 or L2
norm_type = 1 if norm == "L1" else 2
# Uses pixel values to classify images
if __name__ == "__main__":
    print("KNearestNeighbor.py")
    training_data, training_labels, testing_data, testing_labels, labels = LoadImages.all_images()
    figure, axis = plt.subplots(1, 4, squeeze=False)
    axis[0, 0].set_xlabel("Iterations")
    axis[0, 0].set_ylabel("Accuracy")
    iteration = 0
    print(f"Norm: {norm}")
    for i in range(3, 11, 2):
        K = i
        print(f"K: {K}")
        accuracy = calculate(training_data, training_labels, testing_data, testing_labels)
        axis[0, iteration].plot(accuracy)
        axis[0, iteration].set_xlim([1, range(len(testing_data))])
        axis[0, iteration].set_ylim([0, 1])
        axis[0, iteration].set_title(f"K: {K} | {norm}")
        iteration += 1
    plt.savefig(f"../Output Pictures/Nearest Neighbors {norm}.png")
    plt.show()
    plt.close()
