from argparse import ArgumentParser
from math import ceil
from os import listdir, remove, makedirs, system
from os.path import join, getsize, exists
from shutil import move
from time import sleep

from moviepy.video.io.VideoFileClip import VideoFileClip

from config import Config


def get_next_interval(duration, num_splits):
    segment_duration = duration / num_splits
    for i in range(num_splits):
        yield i * segment_duration, (i + 1) * segment_duration


def upload_file(file_path, tag_list, arweave_key):
    tags_str = ' '.join([f'--tag {tag}' for tag in tag_list])
    cmd = f'arweave deploy {file_path} --key-file {arweave_key} ' \
          f'--force-skip-confirmation --force-skip-warnings {tags_str}'
    print('The command with which the video is uploaded:')
    print(cmd)
    system(cmd)


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
                    tags.append('Feed-Name:VideoPresenceTracker')

                    identities_file_path = join(conf.VIDEO_DIR, f'{file_}_identities')
                    with open(identities_file_path, 'r') as identities_file:
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
                            str(upload_file(file_to_upload, tags, args.arweave_key))
                            if 'part' in file_to_upload:
                                print(f'Deleting video part {file_to_upload}.')
                                remove(file_to_upload)
                                to_move = f'{file_to_upload[0:-10]}.mp4'
                                if exists(to_move):
                                    move(to_move, uploaded_dir)
                            else:
                                move(file_to_upload, uploaded_dir)
                    # Move the identities file
                    move(identities_file_path, uploaded_dir)

            print(f'Going to sleep for {conf.SLEEP_INTERVAL} seconds')
            sleep(conf.SLEEP_INTERVAL)
        except KeyboardInterrupt:
            print('Exiting')
            break
