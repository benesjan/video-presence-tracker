"""
A script which computes the reference feature vectors from the images
within the dataset and saves them along with labels to a pickle file.
"""
from os import listdir
from os.path import join

import cv2
import numpy as np

from classifierwrapper import ClassifierWrapper
from config import Config
from pickle_utils import save_pickle

if __name__ == '__main__':
    # 1) Load configuration and model
    conf = Config()
    labels, features = [], []
    classifier_wrapper = ClassifierWrapper(conf.MODEL_WEIGHTS_PATH, labels, features)

    # 2) Iterate through the dataset folders (the names of those folders are labels)
    for name in listdir(conf.DATASET):
        name_features = []
        # 3) Iterate through the images within the folder
        for image_name in listdir(join(conf.DATASET, name)):
            # 4) Load the image and compute the features
            image_path = join(conf.DATASET, name, image_name)
            image = cv2.imread(image_path)
            image_features = classifier_wrapper.get_features(image)

            if len(image_features) == 0:
                print(f'No faces detected in the image on path {image_path}')
            elif image_features.shape[0] > 1:
                print(f'Multiple faces detected in the image on path {image_path}.'
                      f'Cannot distinguish which face belongs to {name}')
            else:
                name_features.append(image_features)

        # 5) Compute representative feature by averaging all the corresponding ones
        name_features = np.vstack(name_features)
        representative_feature = np.sum(name_features, axis=0) / name_features.shape[0]

        # 6) Append label and feature to the corresponding lists
        labels.append(name)
        features.append(representative_feature)

    # 7) Convert list of arrays to an array where each row is one feature vector
    features = np.vstack(features)

    # 8) Save the data as a pickle file
    save_pickle(conf.REPRESENTATIONS, [labels, features])
