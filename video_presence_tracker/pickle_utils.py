import pickle


def save_pickle(file_path, object_to_save):
    """
    A function which saves object to pickle file
    :param file_path: a path to the pickle file
    :param object_to_save: an object to save
    :return: None
    """
    with open(file_path, 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    """
    A function which returns the object stored in the pickle file
    :param file_path: a path to the pickle file
    :return: the object stored within the pickle file
    """
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
