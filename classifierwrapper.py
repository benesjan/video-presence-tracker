import cv2
import numpy as np
import torch
from mtcnn import detect_faces
from scipy.spatial.distance import cdist
from skimage import transform as trans
from torch.nn import DataParallel

from models.resnet import resnet_face18


class ClassifierWrapper:
    """
    A class implementing the classification pipeline
    """

    def __init__(self, model_weights_path, ref_labels, ref_features, threshold=0.65):
        """
        :param model_weights_path: str, path to the model weights
        :param ref_labels: a list containing the labels,
               the index corresponds to the row withing ref_features
        :param ref_features: numpy.ndarray where each row is 1 feature vector
        :param threshold: a float representing the maximum cosine distance
               between reference feature vector and a feature vector
               belonging to the same class
        """
        self.ref_labels = ref_labels
        self.ref_features = ref_features
        self.torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold = threshold
        self._set_model(model_weights_path)

    def _set_model(self, model_weights_path):
        """
        A function which instantiates the model and loads the weights
        :param model_weights_path: str, path to the model weights
        :return: None
        """
        model = resnet_face18(False)
        model = DataParallel(model)
        model.load_state_dict(torch.load(model_weights_path, map_location=self.torch_device))
        model.to(self.torch_device)
        model.eval()
        self.model = model

    @staticmethod
    def _frontalize_face(img, landmarks):
        """
        A function which returns 128x128 image with frontalized face
        :param img: image with face
        :param landmarks: numpy.ndarray
        :return: numpy.ndarray, image with frontalized face
        """
        target_res = (128, 128)

        # Convert landmarks of shape (1, 10) to array of coordinates of 5 facial points (shape (5, 2))
        dst = landmarks.astype(np.float32)
        facial5points = np.array([[dst[j], dst[j + 5]] for j in range(5)])

        # The desired position of landmarks
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        src[:, 0] *= (target_res[0] / 96)
        src[:, 1] *= (target_res[1] / 112)

        # Compute the parameters of the affine transformation
        tform = trans.SimilarityTransform()
        tform.estimate(facial5points, src)
        M = tform.params[0:2, :]

        # Applying the transformation matrix M to the original image
        img_warped = cv2.warpAffine(img, M, target_res, borderValue=0.0)

        return img_warped

    @staticmethod
    def _process_image(img):
        """
        A function, which transforms the image into a form which can be fed to the model
        (originally from arcface-pytorch project)
        :param img: numpy.ndarray, the image to process
        :return: numpy.ndarray, the processed image
        """
        # Stack image and it's flipped version. Output dimensions: (128, 128, 2)
        img = np.dstack((img, np.fliplr(img)))
        # Transpose. Output dimensions: (2, 128, 128)
        img = img.transpose((2, 0, 1))
        # Add dimension. Output dimensions: (2, 1, 128, 128)
        img = img[:, np.newaxis, :, :]
        img = img.astype(np.float32, copy=False)
        # Normalize to <-1, 1>
        img -= 127.5
        img /= 127.5
        return img

    def _get_features_for_batch(self, imgs):
        """
        Function which returns features for one batch of images stacked on top of each other
        (originally from arcface-pytorch project)
        :param imgs: numpy.ndarray, an array of images, dimensions: (num_img * 2, 1, 128, 128)
        :return: numpy.ndarray, an array of features, dimensions: (num_img, 1024)
        """
        data = torch.from_numpy(imgs)
        data = data.to(self.torch_device)
        output = self.model(data)
        output = output.data.cpu().numpy()

        fe_1 = output[::2]
        fe_2 = output[1::2]
        features = np.hstack((fe_1, fe_2))

        return features

    def get_features(self, img):
        """
        A function which detects faces in the image, and computes the feature vector for each detection
        :param img: numpy.ndarray, dimensions: (height, weight, 3)
        :return: numpy.ndarray, an array of features, dimensions: (num_img, 1024)
        """
        features = []
        try:
            bboxes, landmarks = detect_faces(img)
            if len(bboxes) != 0:
                faces = []
                for box_landmarks in landmarks:
                    if img.ndim == 3:
                        # For some reason, the image sometimes contains all the colors and sometimes not -->
                        # doesn't matter as the classifier expects 2D image anyway
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    face_img = self._frontalize_face(img, box_landmarks)
                    face_img = self._process_image(face_img)
                    faces.append(face_img)
                faces = np.vstack(faces)
                features = self._get_features_for_batch(faces)
        except Exception as err:
            print(f'\033[93mException: {err} --> skipping the frame classification\033[0m')
        return features

    def get_labels(self, img):
        """
        A function which returns labels for the image.
        :param img: numpy.ndarray, dimensions: (height, weight, 3)
        :return: a set of labels
        """
        detected_labels = set()
        features = self.get_features(img)
        if len(features) == 0:
            return detected_labels

        # Classification
        dists = cdist(features, self.ref_features, metric='cosine')
        decisions = dists < self.threshold
        presence = np.sum(decisions, axis=0) >= 1
        label_indices = np.where(presence)[0]
        for label_i in label_indices:
            detected_labels.add(self.ref_labels[label_i])

        return detected_labels
