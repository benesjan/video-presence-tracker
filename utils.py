import pickle

import torch
from torch.nn import DataParallel

from models.resnet import resnet_face18


def load_model(config):
    model = resnet_face18(False)
    model = DataParallel(model)

    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))
    model.to(config.DEVICE)

    model.eval()
    return model


def save_pickle(file_path, object_to_save):
    with open(file_path, 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
