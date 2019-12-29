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

    classifier = ClassifierWrapper(conf.MODEL_PATH, labels, features)

    # 2) Compute representative feature vectors from the dataset
    for name in listdir(conf.DATASET):
        name_features = []
        for image_name in listdir(join(conf.DATASET, name)):
            image_path = join(conf.DATASET, name, image_name)
            image = cv2.imread(image_path)

            image_features = classifier.get_features(image)

            if len(image_features) == 0:
                print(f'No faces detected in the image on path {image_path}')
            elif image_features.shape[0] > 1:
                print(f'Multiple faces detected in the image on path {image_path}.'
                      f'Cannot distinguish which face belongs to {name}')
            else:
                name_features.append(image_features)

        name_features = np.vstack(name_features)
        representative_feature = np.sum(name_features, axis=0) / name_features.shape[0]

        labels.append(name)
        features.append(representative_feature)

    features = np.vstack(features)

    save_pickle(conf.REPRESENTATIONS, [labels, features])
