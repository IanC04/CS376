import numpy as np
import LoadImages
import KNearestNeighbor
import AdaBoost
import matplotlib.pyplot as plt


def n_fold_cross_validation(all_data: np.ndarray, all_labels: np.ndarray, slice: int, n: int = 5):
    chunk_size = int(len(all_data) / n)
    train_data_1 = all_data[:chunk_size * slice]
    train_data_2 = all_data[chunk_size * (slice + 1):]
    train_data = np.concatenate((train_data_1, train_data_2))

    train_labels_1 = all_labels[:chunk_size * slice]
    train_labels_2 = all_labels[chunk_size * (slice + 1):]
    train_labels = np.concatenate((train_labels_1, train_labels_2))

    test_data = all_data[chunk_size * slice: chunk_size * (slice + 1)]
    test_labels = all_labels[chunk_size * slice: chunk_size * (slice + 1)]
    return train_data, train_labels, test_data, test_labels


def compute_best_k(all_data, all_labels, label_names):
    n = 5
    accuracies = np.zeros(5)
    for i, k in enumerate(range(1, 11, 2)):
        accuracy_k = 0
        for j in range(n):
            # 5-fold cross-validation
            tr_d, tr_l, te_d, te_l = n_fold_cross_validation(all_data, all_labels, j)
            accuracy = KNearestNeighbor.computeKNN(k, 1, tr_d, tr_l, te_d, te_l, label_names)
            accuracy = accuracy[-1]
            print(accuracy)
            accuracy_k += accuracy
        accuracies[i] = accuracy_k / n
    print(accuracies)


def compute_best_weak_classifiers():
    pass

def confusion_matrix_knn(training_data, training_labels, testing_data, testing_labels, label_names):
    k = 1
    accuracy, predicted = KNearestNeighbor.computeKNN(k, 1, training_data, training_labels, testing_data, testing_labels, label_names)
    confusion_matrix = np.zeros((len(label_names), len(label_names)), dtype=np.uint32)
    for i, label in enumerate(testing_labels):
        confusion_matrix[label, predicted[i]] += 1
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(confusion_matrix)
    for (i, j), z in np.ndenumerate(confusion_matrix):
        ax.text(j, i, '{:d}'.format(z), ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    fig.colorbar(cax)

    ax.set_xticks(np.arange(len(label_names)))
    ax.set_yticks(np.arange(len(label_names)))
    fig.suptitle('Predicted')
    ax.set_xticklabels([s for s in label_names])
    ax.set_yticklabels(["Expected " + s for s in label_names])

    plt.savefig(f"../Output Pictures/Confusion Matrix KNN.png")
    plt.show()
    plt.close()


if __name__ == "__main__":
    tr_d, tr_l, te_d, te_l, l_names = LoadImages.all_images()

    all_d = np.concatenate((tr_d, te_d))
    all_l = np.concatenate((tr_l, te_l))
    # compute_best_k(all_d, all_l, l_names)
    confusion_matrix_knn(tr_d, tr_l, te_d, te_l, l_names)