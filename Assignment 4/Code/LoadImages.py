directory = f"../Assignment 4 Pics/cifar-10-python/cifar-10-batches-py"
files = [f"{directory}/data_batch_{i}" for i in range(1, 6)]


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == "__main__":
    batches = tuple(unpickle(file) for file in files)
    print(batches)
