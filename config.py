from os import path


class Config:

    def __init__(self):
        self.PROJECT_ROOT = path.dirname(path.abspath(__file__))

        # Analyze every n-th frame
        self.NTH_FRAME = 30
        # Check if there was target person detected in the last m analyses - if yes the recording continues
        self.RECORD_IF_IN_M_ANAL = 15

        self.MODEL_PATH = f'{self.PROJECT_ROOT}/model_weights/resnet18_110.pth'

        self.DATASET = f'{self.PROJECT_ROOT}/dataset'

        self.REPRESENTATIONS = f'{self.PROJECT_ROOT}/representations.pickle'

        self.VIDEO_DIR = f'{self.PROJECT_ROOT}/video_out'