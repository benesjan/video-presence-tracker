from argparse import ArgumentParser

from moviepy.editor import *
from pafy import pafy

from classifierwrapper import ClassifierWrapper
from config import Config
from pickle_utils import load_pickle
from videoprocessor import VideoProcessor

if __name__ == '__main__':
    parser = ArgumentParser(description='A bot saving video clips from YouTube video stream'
                                        'containing people whose faces are in the dataset.')
    parser.add_argument('--url', help='URL of the video stream.', required=True, type=str)
    parser.add_argument("--display-video", default=False, action="store_true",
                        help="Pass this flag as argument to display the video while processing.")
    args = parser.parse_args()

    # Load configuration and model
    conf = Config()

    # Load reference features and labels.
    ref_labels, ref_features = load_pickle(conf.REPRESENTATIONS)
    classifier_wrapper = ClassifierWrapper(conf.MODEL_PATH, ref_labels, ref_features)
    video_processor = VideoProcessor(ref_labels, ref_features, classifier_wrapper, conf.VIDEO_DIR, args.display_video)

    # Get the video stream and pass it to OpenCV
    # video = pafy.new(args.url)
    video = pafy.new('https://www.youtube.com/watch?v=K_IR90FthXQ')
    stream = video.getbest(preftype='mp4')

    video = VideoFileClip(stream.url)

    video_processor.process(video, conf.NTH_FRAME, conf.RECORD_IF_IN_M_ANAL)
