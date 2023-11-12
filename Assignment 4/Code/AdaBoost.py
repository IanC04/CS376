import LoadImages
import numpy as np
import matplotlib.pyplot as plt

def classify():
    pass

def train():
    pass

def test():
    pass

classifier = None
# Uses weak classifiers to classify images
if __name__ == "__main__":
    print("AdaBoost.py")
    training, testing, labels = LoadImages.all_images()
    classifier = np.zeros((labels[b'num_vis'], 2))
    pass