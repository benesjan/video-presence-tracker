"""
This script fetches a path to video as an argument and processes the specified video.
"""
from argparse import ArgumentParser

from moviepy.video.io.VideoFileClip import VideoFileClip

from config import Config
from video_presence_tracker import load_pickle, VideoProcessor

if __name__ == '__main__':
    parser = ArgumentParser(description='A script which cuts out video segments'
                                        'containing people whose faces are in the dataset.')
    parser.add_argument('--video-path', help='Path to the source video.', required=True, type=str)
    parser.add_argument("--display-video", default=False, action="store_true",
                        help="Pass this flag as argument to display the video while processing.")
    args = parser.parse_args()

    # Load configuration
    conf = Config()

    # Load reference features and labels.
    ref_labels, ref_features = load_pickle(conf.REPRESENTATIONS)

    # Instantiate the video processor
    video_processor = VideoProcessor(ref_labels, ref_features, conf.MODEL_WEIGHTS_PATH,
                                     conf.VIDEO_DIR, args.display_video)

    video = VideoFileClip(args.video_path)
    try:
        print(f'Processing the video.')
        video_processor.process(video, conf.NTH_FRAME, conf.RECORD_IF_IN_M_ANAL)
    except KeyboardInterrupt:
        print('Exiting')
