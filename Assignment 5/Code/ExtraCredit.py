import os

from sklearn.ensemble import AdaBoostClassifier

import numpy as np
from matplotlib import pyplot as plt
import timeit


def precision_recall_curve(custom_model, testing_set):
    faces = custom_model.testing_faces
    non_faces = custom_model.testing_non_faces
    # haar_cascade = cv2.CascadeClassifier('../Assignment 5
    # Pics/haarcascade_frontalface_default.xml')
    haar_cascade = AdaBoostClassifier(n_estimators=20, random_state=0)

    # for img in original_testing_imgs:
    #     img = img.copy()
    #     gray_img = AdaBoost.rgb2gray(img).astype(np.uint8)
    #     faces_rect = haar_cascade.detectMultiScale(gray_img, scaleFactor=1.05, minSize=(24, 24),
    #                                                minNeighbors=6)
    #     for (x, y, w, h) in faces_rect:
    #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
    #     plt.imshow(img)
    #     plt.waitforbuttonpress()
    #     plt.close()

    precisions, recalls = [], []
    x = np.concatenate((custom_model.testing_faces,
                        custom_model.testing_non_faces))
    y = np.concatenate((np.ones(len(faces)), np.zeros(len(non_faces))))

    x_haar = np.concatenate((custom_model.training_face_haar_features,
                             custom_model.training_non_face_haar_features))
    y_haar = np.concatenate((np.ones(len(custom_model.training_face_haar_features)),
                             np.zeros(len(custom_model.training_non_face_haar_features))))

    import pickle
    import LoadImages
    if os.path.exists(f'{LoadImages.CACHE_PATH}/haar_cascade.pkl'):
        print('Loading Haar Cascade from cache')
        haar_cascade = pickle.load(open(f'{LoadImages.CACHE_PATH}/haar_cascade.pkl', 'rb'))
    else:
        print('Training Haar Cascade')
        start_train_time = timeit.default_timer()
        haar_cascade.fit(x_haar, y_haar)
        end_train_time = timeit.default_timer()
        # print(f"Training elapsed time: {end_train_time - start_train_time}")
        pickle.dump(haar_cascade, open(f'{LoadImages.CACHE_PATH}/haar_cascade.pkl', 'wb'))
    del pickle
    del LoadImages

    average_face = AdaBoost.average_images(faces)
    average_non_face = AdaBoost.average_images(non_faces)
    # plt.imshow(average_face)
    # plt.title('Average Face')
    # plt.waitforbuttonpress()
    # plt.imshow(average_non_face)
    # plt.title('Average Non-Face')
    # plt.waitforbuttonpress()

    for scale in range(1, 21):
        FP, TP, FN, TN = 0, 0, 0, 0

        start_test_time = timeit.default_timer()
        for idx, img in enumerate(x):
            img_haar = custom_model.testing_face_haar_features[idx] if idx < len(faces) else \
                custom_model.testing_non_face_haar_features[idx - len(faces)]

            face_prob = haar_cascade.predict_proba(np.array([img_haar]))
            close_to_face = np.sum(np.sum(np.abs(img - average_face), axis=0), axis=0)
            close_to_non_face = np.sum(np.sum(np.abs(img - average_non_face), axis=0), axis=0)
            closer_to_face = close_to_face < close_to_non_face
            bias = 1.5 if closer_to_face else -1.5

            has_face = face_prob[0][1] * bias > scale * 0.05
            if has_face:
                if y[idx] == 1:
                    TP += 1
                else:
                    FP += 1
            else:
                if y[idx] == 0:
                    TN += 1
                else:
                    FN += 1
        end_test_time = timeit.default_timer()
        # print(f"Testing elapsed time: {end_test_time - start_test_time}")
        precision = 0 if TP == 0 else TP / (TP + FP)
        recall = TP / (TP + FN)
        recalls.append(recall)
        precisions.append(precision)

    plt.plot(recalls, precisions)
    # plt.title('Precision Recall Curve')
    plt.title('Precision Recall Curve with Bias')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.waitforbuttonpress()


if __name__ == "__main__":
    from AdaBoost import AdaBoost
    from AdaBoost import DecisionStump

    custom_model = AdaBoost.load_model()
    del DecisionStump

    original_training_imgs = list()
    for layer in custom_model.original_training_images:
        for img in layer:
            original_training_imgs.append(layer[img])

    original_testing_imgs = list()
    for layer in custom_model.original_testing_images:
        for img in layer:
            original_testing_imgs.append(layer[img])

    precision_recall_curve(custom_model, original_testing_imgs)
