import base64
from argparse import ArgumentParser
from math import ceil
from os import listdir, remove, makedirs
from os.path import join, getsize, exists
from shutil import move
from time import sleep

import requests
from moviepy.video.io.VideoFileClip import VideoFileClip

from config import Config


def get_next_interval(duration, num_splits):
    segment_duration = duration / num_splits
    for i in range(num_splits):
        yield i * segment_duration, (i + 1) * segment_duration


def upload_and_get_url(file_path, tags):
    with open(file_path, 'rb') as video_file:
        file_base64 = base64.b64encode(video_file.read())
    r = requests.post('http://localhost:1908/raw', data={'data': file_base64, 'tags': tags})
    return r.content


if __name__ == '__main__':
    # Load configuration
    conf = Config()

    parser = ArgumentParser(description='A script which periodically checks for new videos'
                                        f'in {conf.VIDEO_DIR} and uploads them to the permaweb')
    parser.add_argument('--arweave-key', help='Arweave key (json file).', required=True, type=str)
    args = parser.parse_args()

    uploaded_dir = join(conf.VIDEO_DIR, 'uploaded')
    if not exists(uploaded_dir):
        makedirs(uploaded_dir)

    while True:
        try:
            for file_ in listdir(conf.VIDEO_DIR):
                if file_.endswith('mp4'):
                    files_to_upload, tags = [], []
                    tags.append({'Feed-Name': 'VideoPresenceTracker'})

                    with open(join(conf.VIDEO_DIR, f'{file_}_identities'), 'r') as identities_file:
                        identities = identities_file.read().split(',')

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
                    with open(conf.TRANSACTION_LOG, 'a') as f:
                        for file_to_upload in files_to_upload:
                            f.write(str(upload_and_get_url(file_to_upload, tags)))
                            f.write('\n')
                            if 'part' in file_to_upload:
                                print(f'Deleting video part {file_to_upload}.')
                                remove(file_to_upload)
                                to_move = f'{file_to_upload[0:-10]}.mp4'
                                if exists(to_move):
                                    move(to_move, uploaded_dir)
                            else:
                                move(file_to_upload, uploaded_dir)

            print(f'Going to sleep for {conf.SLEEP_INTERVAL} seconds')
            sleep(conf.SLEEP_INTERVAL)
        except KeyboardInterrupt:
            print('Exiting')
            break
