"""
Written by Ian Chen on 11/25/2023
GitHub: https://github.com/IanC04
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

FOLDS_PATH = f"../Assignment 5 Pics/Data Set/FDDB-folds"
PICTURES_PATH = f"../Assignment 5 Pics/Data Set/originalPics"
CACHE_PATH = f"../Custom Cache"
FACE_KEYS = ("major_axis_radius", "minor_axis_radius", "angle", "center_x", "center_y", "1")


def load_folds():
    """
    Load folds from fold_path or from cache if loaded previously
    """
    if os.path.exists(f"{CACHE_PATH}/folds.npy"):
        with open(f"{CACHE_PATH}/folds.npy", "rb") as f:
            print("Cache found. Loading folds...")
            training_folds = np.load(f, allow_pickle=True)
            testing_folds = np.load(f, allow_pickle=True)
        return training_folds, testing_folds
    else:
        print("Cache not found. Generating folds...")
        folds = list()

        # Ten folds
        for i in tqdm(range(1, 11), desc="Loading folds"):
            images = f"{FOLDS_PATH}/FDDB-fold-{i:02d}.txt"
            ellipse = f"{FOLDS_PATH}/FDDB-fold-{i:02d}-ellipseList.txt"

            folds_data = dict()
            with open(images) as f:
                with open(ellipse) as e:
                    for img in f:
                        img = img.strip()
                        assert img not in folds_data, f"Image already in folds_data: {img}"
                        folds_data[img] = list()

                        img_e = e.readline()
                        img_e = img_e.strip()
                        assert img == img_e, f"Image name mismatch: {img} != {img_e}"
                        num_faces = int(e.readline())

                        # Multiple faces per image
                        for j in range(num_faces):
                            face_data = e.readline().split()
                            face_data.pop()
                            face_data = [float(x) for x in face_data]
                            face_data = get_bbox(face_data)
                            folds_data[img].append(np.array(face_data))
                        folds_data[img] = np.array(folds_data[img])
            folds.append(folds_data)
        folds = np.array(folds)
        training_folds = folds[:8]
        testing_folds = folds[8:]
        with open(f"{CACHE_PATH}/folds.npy", "wb") as f:
            np.save(f, training_folds)
            np.save(f, testing_folds)
        return training_folds, testing_folds


def get_bbox(face_data):
    """
    Convert ellipse data to bounding box data
    """
    major, minor, angle, x, y = face_data
    # So no division by zero
    if abs(angle) < 1e-8:
        angle = 1e-8
    assert angle >= -np.pi / 2 and angle <= np.pi / 2, f"Angle out of range: {angle}"

    t = np.arctan(-minor * np.tan(angle) / major)
    [x1, x2] = sorted([x + major * np.cos(t) * np.cos(angle) - minor * np.sin(t) * np.sin(angle)
                       for t in (t + np.pi, t)])
    t = np.arctan(minor * (1 / np.tan(angle)) / major)
    [y1, y2] = sorted([y + minor * np.sin(t) * np.cos(angle) + major * np.cos(t) * np.sin(angle)
                       for t in (t + np.pi, t)])
    return [x1, y1, x2, y2]


def load_images(folds):
    """
    Load images from pictures_path or from cache if loaded previously
    """
    if os.path.exists(f"{CACHE_PATH}/images.npy"):
        with open(f"{CACHE_PATH}/images.npy", "rb") as f:
            print("Cache found. Loading images...")
            training_images = np.load(f, allow_pickle=True)
            testing_images = np.load(f, allow_pickle=True)
        return training_images, testing_images
    else:
        print("Cache not found. Generating images...")
        images = list()
        for fold in tqdm(folds, desc="Loading images"):
            images_in_fold = dict()
            for path in fold:
                img_path = f"{PICTURES_PATH}/{path}.jpg"
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                images_in_fold[path] = np.array(img)
            images.append(images_in_fold)
        images = np.array(images)
        training_images = images[:8]
        testing_images = images[8:]
        with open(f"{CACHE_PATH}/images.npy", "wb") as f:
            np.save(f, training_images)
            np.save(f, testing_images)
        return training_images, testing_images


def show_bbox(folds, images, paths=None):
    """
    Show bounding boxes on images
    """
    if paths is None:
        for index in tqdm(range(folds.size), desc="Overlaying bounding boxes"):
            for path in folds[index]:
                img = images[index][path]
                for face in folds[index][path]:
                    x1, y1, x2, y2 = np.round(face).astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                plt.imshow(img)
                plt.waitforbuttonpress()
                plt.close()
    else:
        for index in tqdm(range(folds.size), desc="Overlaying bounding boxes for specific images"):
            for path in paths:
                img = images[index][path]
                for face in folds[index][path]:
                    x1, y1, x2, y2 = np.round(face).astype(int)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                plt.imshow(img)
                plt.waitforbuttonpress()
                plt.close()


if __name__ == "__main__":
    tr_folds, te_folds = load_folds()
    tr_images, te_images = load_images(np.concatenate((tr_folds, te_folds)))
    print("Retrieved all FDDB data.")
    show_bbox(np.concatenate((tr_folds, te_folds)), np.concatenate((tr_images, te_images)),
              paths=None)
