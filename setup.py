from os import listdir
from os.path import join
import numpy as np
import cv2
from mtcnn import detect_faces

from config import Config
from utils import load_model, frontalize_face, process_face, predict

if __name__ == '__main__':
    # 1) Load configuration and model
    conf = Config()
    model = load_model(conf)

    # 2) Compute representative feature vectors from the dataset
    with open(conf.REPRESENTATIONS, 'w') as f:
        for name in listdir(conf.DATASET):
            faces = []
            for image_name in listdir(join(conf.DATASET, name)):
                image_path = join(conf.DATASET, name, image_name)
                image = cv2.imread(image_path)
                bboxes, landmarks = detect_faces(image)
                # Convert image to grayscale
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                if len(bboxes) == 0:
                    print(f'No faces detected in the image on path {image_path}')
                elif len(bboxes) > 1:
                    print(f'Multiple faces detected in the image on path {image_path}.'
                          f'Cannot distinguish which face belongs to {name}')
                else:
                    face_img = frontalize_face(image, landmarks[0])
                    face_img = process_face(face_img)
                    faces.append(face_img)
            features = predict(model, faces)
            representative_feature = np.sum(features, axis=0) / features.shape[0]
            f.write(f'{name}\t{representative_feature.tostring()}\n')
