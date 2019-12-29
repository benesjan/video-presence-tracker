import pickle


def save_pickle(file_path, object_to_save):
    with open(file_path, 'wb') as handle:
        pickle.dump(object_to_save, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(file_path):
    with open(file_path, 'rb') as handle:
        return pickle.load(handle)
