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
    return (training, testing, labels)


if __name__ == "__main__":
    training, testing, labels  = all_images()
    print(f"Training: {training}")
    print(f"Testing: {testing}")
    print(f"Labels: {labels}")
