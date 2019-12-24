import cv2
import numpy as np
import torch
from mtcnn import detect_faces
from scipy.spatial.distance import cdist
from skimage import transform as trans


class ClassifierWrapper:

    def __init__(self, model, ref_labels, ref_features, torch_device, threshold=0.65):
        self.model = model
        self.ref_labels = ref_labels
        self.ref_features = ref_features
        self.torch_device = torch_device
        self.threshold = threshold

    @staticmethod
    def _frontalize_face(img, landmarks):
        target_res = (128, 128)

        # Convert landmarks of shape (1, 10) to array of coordinates of 5 facial points (shape (5, 2))
        dst = landmarks.astype(np.float32)
        facial5points = np.array([[dst[j], dst[j + 5]] for j in range(5)])

        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)

        src[:, 0] *= (target_res[0] / 96)
        src[:, 1] *= (target_res[1] / 112)

        tform = trans.SimilarityTransform()
        tform.estimate(facial5points, src)
        M = tform.params[0:2, :]

        # Applying the transformation matrix M to the original image
        img_warped = cv2.warpAffine(img, M, target_res, borderValue=0.0)

        return img_warped

    # Process image so that it can be fed to NN
    @staticmethod
    def _process_face(im):
        # The following lines are copied from arcface-pytorch project
        # Stack image and it's flipped version. Output dimensions: (128, 128, 2)
        im = np.dstack((im, np.fliplr(im)))
        # Transpose. Output dimensions: (2, 128, 128)
        im = im.transpose((2, 0, 1))
        # Add dimension. Output dimensions: (2, 1, 128, 128)
        im = im[:, np.newaxis, :, :]
        im = im.astype(np.float32, copy=False)
        # Normalize to <-1, 1>
        im -= 127.5
        im /= 127.5
        return im

    # Get features for 1 batch
    def _predict(self, images):
        images_array = np.vstack(images)
        data = torch.from_numpy(images_array)
        data = data.to(self.torch_device)
        output = self.model(data)
        output = output.data.cpu().numpy()

        fe_1 = output[::2]
        fe_2 = output[1::2]
        features = np.hstack((fe_1, fe_2))

        return features

    def get_features(self, img):
        features = []
        try:
            bboxes, landmarks = detect_faces(img)
            if len(bboxes) != 0:
                faces = []
                for box_landmarks in landmarks:
                    if img.ndim != 3:
                        print(f'Incorrect number of channels: {img.ndim} instead of 3 --> Skipping the image')
                        break
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    face_img = self._frontalize_face(img, box_landmarks)
                    face_img = self._process_face(face_img)
                    faces.append(face_img)
                features = self._predict(faces)
        except Exception as err:
            print(f'Exception: {err}')
        return features

    def get_labels(self, img):
        detected_labels = []
        features = self.get_features(img)
        if len(features) == 0:
            return detected_labels

        # Classification
        dists = cdist(features, self.ref_features, metric='cosine')
        decisions = dists < self.threshold
        presence = np.sum(decisions, axis=0) >= 1
        label_indices = np.where(presence)[0]
        for label_i in label_indices:
            detected_labels.append(self.ref_labels[label_i])

        return detected_labels
