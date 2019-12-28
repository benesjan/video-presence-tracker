from argparse import ArgumentParser
from os.path import join
from time import time
from moviepy.editor import *

import cv2
from pafy import pafy

from classifierwrapper import ClassifierWrapper
from config import Config
from utils import load_model, load_pickle

if __name__ == '__main__':
    parser = ArgumentParser(description='A bot saving video clips from YouTube video stream'
                                        'containing people whose faces are in the dataset.')
    parser.add_argument('--url', help='URL of the video stream.', required=True, type=str)
    args = parser.parse_args()

    display_video = True

    # Load configuration and model
    conf = Config()

    # Load reference features and labels.
    ref_labels, ref_features = load_pickle(conf.REPRESENTATIONS)
    classifier = ClassifierWrapper(load_model(conf), ref_labels, ref_features, conf.DEVICE)

    # Get the video stream and pass it to OpenCV
    # video = pafy.new(args.url)
    video = pafy.new('https://www.youtube.com/watch?v=K_IR90FthXQ')
    stream = video.getbest(preftype='mp4')

    video = VideoFileClip(stream.url)
    audio = video.audio

    # A list of boolean values which represent whether there was somebody from reference dataset detected
    presence_of_reference = [False for i in range(conf.check_every_m_analyses)]

    video_name, currently_detected, N_counter, recording_t = None, set(), 1, 0
    for t, video_frame in video.iter_frames(with_times=True):

        if N_counter == conf.n_th_frame:
            detected_labels = classifier.get_labels(video_frame)

            if len(detected_labels) != 0:
                presence_of_reference.append(True)
                currently_detected.update(detected_labels)
                print(f'Identities of interest currently present in the recorded clip: {currently_detected}')
            else:
                presence_of_reference.append(False)

            # Remove the oldest detection
            presence_of_reference = presence_of_reference[1:]

            if True in presence_of_reference:
                if video_name is None:
                    print('Recording started')
                    # Instantiate the video writer
                    frame_width, frame_height = video.size
                    video_name = f'{time()}.mp4'
                    recording_t = t

            elif video_name is not None:
                print('Recording stopped')
                video_clip = video.subclip(recording_t, t)
                video_clip.write_videofile(join(conf.VIDEO_OUT, video_name))
                with open(join(conf.VIDEO_OUT, f'{video_name}_identities'), 'a') as f:
                    f.write(",".join(currently_detected))
                video_name = None

            N_counter = 1
        else:
            N_counter += 1

        if display_video:
            # OpenCV expects frame in BGR color format
            video_frame_BGR = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
            # Display the resulting frame
            cv2.imshow('frame', video_frame_BGR)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # When everything done, release the capture
    cv2.destroyAllWindows()

    if video_name is not None:
        print('Recording stopped')
        video_clip = video.subclip(recording_t)
        video_clip.write_videofile(join(conf.VIDEO_OUT, video_name))
        with open(join(conf.VIDEO_OUT, f'{video_name}_identities'), 'a') as f:
            f.write(",".join(currently_detected))
