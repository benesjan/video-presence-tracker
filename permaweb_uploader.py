from argparse import ArgumentParser
from math import ceil
from os import listdir
from os.path import join, getsize
from time import sleep

from moviepy.video.io.VideoFileClip import VideoFileClip

from config import Config


def get_next_interval(duration, num_splits):
    segment_duration = duration / num_splits
    for i in range(num_splits):
        yield i * segment_duration, (i + 1) * segment_duration


if __name__ == '__main__':
    # Load configuration
    conf = Config()

    parser = ArgumentParser(description='A script which periodically checks for new videos'
                                        f'in {conf.VIDEO_DIR} and uploads them to the permaweb')
    parser.add_argument('--arweave-key', help='Arweave key (json file).', required=True, type=str)
    args = parser.parse_args()

    while True:
        try:
            files_to_upload = []
            for file_ in listdir(conf.VIDEO_DIR):
                if file_.endswith('mp4'):
                    video_path = join(conf.VIDEO_DIR, file_)
                    size_in_bytes = getsize(video_path)
                    if size_in_bytes > conf.MAX_TRANSACTION_SIZE:
                        num_splits = ceil(size_in_bytes / conf.MAX_TRANSACTION_SIZE)
                        video = VideoFileClip(video_path)
                        for i, (t_start, t_end) in enumerate(get_next_interval(video.duration, num_splits)):
                            video_clip_path = join(conf.VIDEO_DIR, f'{file_[0:-4]}_part{i}.mp4')
                            video_clip = video.subclip(t_start, t_end)
                            video_clip.write_videofile(video_clip_path, codec='libx265')
                            files_to_upload.append(video_clip_path)
                    else:
                        files_to_upload.append(video_path)

            print(files_to_upload)

            print(f'Going to sleep for {conf.SLEEP_INTERVAL} seconds')
            sleep(conf.SLEEP_INTERVAL)
        except KeyboardInterrupt:
            print('Exiting')
            break
