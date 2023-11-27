"""
Written by Ian Chen on 11/25/2023
GitHub: https://github.com/IanC04
"""

import LoadImages

class AdaBoost:
    """
    AdaBoost class
    """
    def __init__(self, folds, images):
        self.folds = folds
        self.images = images

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    folds = LoadImages.load_folds()
    images = LoadImages.load_images(folds)
    model = AdaBoost(folds, images)
    model.train()