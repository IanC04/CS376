import numpy as np

directory = f"../Assignment 4 Pics/cifar-10-python/cifar-10-batches-py"
training_files = [f"{directory}/data_batch_{i}" for i in range(1, 6)]
testing_file = f"{directory}/test_batch"
labels_file = f"{directory}/batches.meta"


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def all_images() -> tuple:
    training = tuple(unpickle(file) for file in training_files)
    testing = unpickle(testing_file)
    labels = unpickle(labels_file)

    training_data = []
    training_labels = []
    testing_data = testing[b'data']
    testing_labels = testing[b'labels']
    for i in range(len(training)):
        training_data.append(training[i][b'data'])
        training_labels.append(training[i][b'labels'])

    training_data = np.concatenate(np.array(training_data))
    training_labels = np.concatenate(np.array(training_labels))
    testing_data = np.array(testing_data)
    testing_labels = np.array(testing_labels)
    labels = np.array(labels[b'label_names'])

    training_data = training_data.astype(np.int32)
    training_labels = training_labels.astype(np.int32)
    testing_data = testing_data.astype(np.int32)
    testing_labels = testing_labels.astype(np.int32)

    labels = np.array(["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
    return (training_data, training_labels, testing_data, testing_labels, labels)


if __name__ == "__main__":
    training, testing, labels = all_images()
    print(f"Training: {training}")
    print(f"Testing: {testing}")
    print(f"Labels: {labels}")
