"""
A script which gets YouTube channel id as an argument from console and
continually processes every new video uploaded to this channel.

Example usage:
$ python main.py --display-video --channel-id UCeY0bbntWzzVIaj2z3QigXg
"""
from argparse import ArgumentParser
from os.path import exists
from time import sleep

import requests
from moviepy.editor import *
from pafy import pafy

from classifierwrapper import ClassifierWrapper
from config import Config
from pickle_utils import load_pickle, save_pickle
from videoprocessor import VideoProcessor


def get_next_new_channel_vid(channel_id, api_key, processed_ids_path):
    """
    A generator function, which returns id and title of not yet processed videos
    :param channel_id: str, YouTube channel ID
    :param api_key: str, YouTube API key
    :param processed_ids_path: path to the pickle file,
           where the processed video ids are stored in a set
    :return: video id and video title
    """
    processed_ids = set()
    if exists(processed_ids_path):
        processed_ids = load_pickle(processed_ids_path)

    params = {
        'channelId': channel_id,
        'key': api_key,
        'part': 'snippet',
        'type': 'video'
    }
    videos = requests.get(url='https://www.googleapis.com/youtube/v3/search', params=params).json()
    for video_dict in videos['items']:
        video_id, title = video_dict['id']['videoId'], video_dict['snippet']['title']
        if video_id not in processed_ids:
            processed_ids.add(video_id)
            yield video_id, title

    save_pickle(processed_ids_path, processed_ids)


if __name__ == '__main__':
    parser = ArgumentParser(description='A bot saving video clips from YouTube video stream'
                                        'containing people whose faces are in the dataset.')
    parser.add_argument('--channel-id', help='URL of the video stream.', required=True, type=str)
    parser.add_argument("--display-video", default=False, action="store_true",
                        help="Pass this flag as argument to display the video while processing.")
    args = parser.parse_args()

    # Load configuration
    conf = Config()

    # Load reference features and labels.
    ref_labels, ref_features = load_pickle(conf.REPRESENTATIONS)

    # Instantiate classifier and video processor
    classifier_wrapper = ClassifierWrapper(conf.MODEL_PATH, ref_labels, ref_features)
    video_processor = VideoProcessor(classifier_wrapper, conf.VIDEO_DIR, args.display_video)

    # Load the YouTube API key
    with open(conf.YOUTUBE_API_KEY_PATH) as f:
        api_key = f.read()

    while True:
        try:
            # Iterate over new videos
            for video_id, video_title in get_next_new_channel_vid(args.channel_id, api_key, conf.PROCESSED_YOUTUBE_IDS):
                # Get the exact URL of the video file
                video = pafy.new(f'https://www.youtube.com/watch?v={video_id}')
                stream = video.getbest(preftype='mp4')

                # Create an instance of moviepy video - used to cut out the snippets containing the target identites
                video = VideoFileClip(stream.url)

                print(f'Processing video with title: {video_title}')
                video_processor.process(video, conf.NTH_FRAME, conf.RECORD_IF_IN_M_ANAL)

            print(f'Going to sleep for {conf.SLEEP_INTERVAL} seconds')
            sleep(conf.SLEEP_INTERVAL)
        except KeyboardInterrupt:
            print('Exiting')
