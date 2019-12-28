from os import path
import torch


class Config:

    def __init__(self):
        self.PROJECT_ROOT = path.dirname(path.abspath(__file__))

        # Analyze every n-th frame
        self.n_th_frame = 30
        # Check if there was target person detected in the last m analyses
        self.check_every_m_analyses = 15

        if torch.cuda.is_available():
            self.DEVICE = torch.device('cuda:0')
        else:
            self.DEVICE = torch.device('cpu')

        self.MODEL_PATH = f'{self.PROJECT_ROOT}/model_weights/resnet18_110.pth'

        self.DATASET = f'{self.PROJECT_ROOT}/dataset'

        self.REPRESENTATIONS = f'{self.PROJECT_ROOT}/representations.pickle'

        self.VIDEO_OUT = f'{self.PROJECT_ROOT}/video_out'