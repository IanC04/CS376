import numpy as np
import LoadImages
import MaxHeap
import matplotlib.pyplot as plt


def euclidean_distance(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Returns the Euclidean distance between two images, square root of each pixel's RGB summed up
    :param img1:
    :param img2:
    :return:
    """
    # L1 Norm: a = np.linalg.norm(img1.astype(np.int32) - img2.astype(np.int32), ord=1)
    # Slower: a = np.sqrt(np.sum(np.square(img1.astype(np.int32)-img2.astype(np.int32))))
    # Faster:  np.linalg.norm(img1.astype(np.int32) - img2.astype(np.int32))
    if norm == "L1":
        return np.linalg.norm(img1.astype(np.int32) - img2.astype(np.int32), ord=1)
    else:
        return np.linalg.norm(img1.astype(np.int32) - img2.astype(np.int32))

def classify(training: np.ndarray, img: np.ndarray) -> int:
    top_matches.clear()
    for batch in training:
        for index, image in enumerate(batch[b'data']):
            distance = euclidean_distance(image, img)
            label = batch[b'labels'][index]
            top_matches.insert((distance, label))
    labels = top_matches.get_labels()
    top = np.amax(labels)
    return top


def calculate(training, testing, subset=100):
    correct = 0
    incorrect = 0
    accuracies = np.zeros(1 + subset)
    for index, image in enumerate(testing[b'data']):
        if index >= subset:
            break
        sol = classify(training, image)
        if sol == testing[b'labels'][index]:
            correct += 1
        else:
            incorrect += 1
        accuracy = correct / (correct + incorrect)
        print(f"Incorrect: {incorrect}, Correct: {correct}, Accuracy: {accuracy}")
        accuracies[1 + index] = accuracy
    return accuracies


K = 7
top_matches = MaxHeap.MaxHeap(K)
subset = 1000
norm = "L2"  # L1 or L2
# Uses pixel values to classify images
if __name__ == "__main__":
    print("KNearestNeighbor.py")
    training, testing, labels = LoadImages.all_images()
    figure, axis = plt.subplots(1, 4, squeeze=False)
    axis[0, 0].set_xlabel("Iterations")
    axis[0, 0].set_ylabel("Accuracy")
    iteration = 0
    print(f"Norm: {norm}")
    for i in range(3, 11, 2):
        K = i
        top_matches = MaxHeap.MaxHeap(K)
        print(f"K: {K}")
        accuracy = calculate(training, testing, subset)
        axis[0, iteration].plot(accuracy)
        axis[0, iteration].set_xlim([1, subset])
        axis[0, iteration].set_ylim([0, 1])
        axis[0, iteration].set_title(f"K: {K} | {norm}")
        iteration += 1
    plt.savefig(f"../Output Pictures/Nearest Neighbors {norm}.png")
    plt.show()
    plt.close()
