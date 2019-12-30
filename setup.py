"""
A script which computes the reference feature vectors from the images
within the dataset and saves them along with labels to a pickle file.
"""
from os import listdir, makedirs
from os.path import join, exists

import cv2
import gdown
import numpy as np

from config import Config
from video_presence_tracker import *

if __name__ == '__main__':
    # 1) Load configuration and model
    conf = Config()
    labels, features = [], []

    # 2) Make sure the app data dir exists
    if not exists(conf.APP_DATA_DIR):
        makedirs(conf.APP_DATA_DIR)

    # 3) Make sure the video dir exists
    if not exists(conf.VIDEO_DIR):
        makedirs(conf.VIDEO_DIR)

    # 4) Check if the model weights were downloaded and do so if not
    if not exists(conf.MODEL_WEIGHTS_PATH):
        print("Downloading the model weights.")
        weights_url = 'https://drive.google.com/uc?id=1wJTbgNT11GSLNZT-nsCzXNeu6Pqh5r3y'
        gdown.download(weights_url, conf.MODEL_WEIGHTS_PATH, quiet=False)

    classifier_wrapper = ClassifierWrapper(labels, features, conf.MODEL_WEIGHTS_PATH)

    # 5) Iterate through the dataset folders (the names of those folders are labels)
    for name in listdir(conf.DATASET):
        name_features = []
        # 6) Iterate through the images within the folder
        for image_name in listdir(join(conf.DATASET, name)):
            # 7) Load the image and compute the features
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

        # 8) Compute representative feature by averaging all the corresponding ones
        name_features = np.vstack(name_features)
        representative_feature = np.sum(name_features, axis=0) / name_features.shape[0]

        # 9) Append label and feature to the corresponding lists
        labels.append(name)
        features.append(representative_feature)

    # 10) Convert list of arrays to an array where each row is one feature vector
    features = np.vstack(features)

    # 11) Save the data as a pickle file
    print('Saving the representation of identities.')
    save_pickle(conf.REPRESENTATIONS, [labels, features])
