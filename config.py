from os import path


class Config:
    """
    A class defining the constants used throughout the project
    """

    def __init__(self):
        self.PROJECT_ROOT = path.dirname(path.abspath(__file__))

        # Analyze every n-th frame in order to decrease computational intensity
        self.NTH_FRAME = 30

        # Check if there was target person detected in the last m analyses - if yes the recording continues
        self.RECORD_IF_IN_M_ANAL = 15

        # A path to model weights
        self.MODEL_PATH = f'{self.PROJECT_ROOT}/model_weights/resnet18_110.pth'

        # A path to the reference dataset - used to compute features of the target identities
        self.DATASET = f'{self.PROJECT_ROOT}/dataset'

        # A path to pickle file where the reference features and corresponding labels will be stored
        self.REPRESENTATIONS = f'{self.PROJECT_ROOT}/representations.pickle'

        # A path to the directory where the processed videos will be stored
        self.VIDEO_DIR = f'{self.PROJECT_ROOT}/video_out'

        # YouTube related
        self.YOUTUBE_API_KEY_PATH = f'{self.PROJECT_ROOT}/api_key'
        self.PROCESSED_YOUTUBE_IDS = f'{self.PROJECT_ROOT}/processed_ids.pickle'

        # The time between new video checks [in seconds]
        self.SLEEP_INTERVAL = 600