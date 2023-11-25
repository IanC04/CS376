"""
Written by Ian Chen on 11/25/2023
GitHub: https://github.com/IanC04
"""
import os
import numpy as np

FOLDS_PATH = f"../Assignment 5 Pics/Data Set/FDDB-folds"
PICTURES_PATH = f"../Assignment 5 Pics/Data Set/originalPics"
CACHE_PATH = f"../Custom Cache"
FACE_KEYS = ("major_axis_radius", "minor_axis_radius", "angle", "center_x", "center_y", "1")


def load_folds():
    """
    Load folds from fold_path
    """
    if os.path.exists(f"{CACHE_PATH}/folds.npy"):
        with open(f"{CACHE_PATH}/folds.npy", "rb") as f:
            folds = np.load(f, allow_pickle=True)
        return folds
    else:
        print("Cache not found. Generating folds...")
        folds = list()

        for i in range(1, 11):
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
                        for j in range(num_faces):
                            face_data = e.readline().split()
                            face_data.pop()
                            face_data = [float(x) for x in face_data]
                            folds_data[img].append(np.array(face_data))
                        folds_data[img] = np.array(folds_data[img])
            folds.append(folds_data)
        folds = np.array(folds)
        with open(f"{CACHE_PATH}/folds.npy", "wb") as f:
            np.save(f, folds)
        return folds


if __name__ == "__main__":
    all_folds = load_folds()

    print("Done.")
