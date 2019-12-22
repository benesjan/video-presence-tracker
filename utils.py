import cv2
import numpy as np
import torch
from skimage import transform as trans
from torch.nn import DataParallel

from config import Config
from models.resnet import resnet_face18


def load_model(config):
    model = resnet_face18(False)
    model = DataParallel(model)

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)

    model.eval()
    return model


def frontalize_face(img, landmarks):
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
def process_face(im):
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
def predict(model, images):
    conf = Config()
    images_array = np.vstack(images)
    data = torch.from_numpy(images_array)
    data = data.to(conf.DEVICE)
    output = model(data)
    output = output.data.cpu().numpy()

    fe_1 = output[::2]
    fe_2 = output[1::2]
    features = np.hstack((fe_1, fe_2))

    return features
