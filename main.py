from argparse import ArgumentParser
from os.path import join
from time import time

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

    # Load configuration and model
    conf = Config()

    # Load reference features and labels.
    ref_labels, ref_features = load_pickle(conf.REPRESENTATIONS)
    classifier = ClassifierWrapper(load_model(conf), ref_labels, ref_features, conf.DEVICE)

    # Get the video stream and pass it to OpenCV
    video = pafy.new(args.url)
    # video = pafy.new('https://www.youtube.com/watch?v=K_IR90FthXQ&t=15s')
    stream = video.getbest(preftype='mp4')
    # stream = video.allstreams[4]
    cap = cv2.VideoCapture()
    cap.open(stream.url)

    try:
        fps = int(video.get(cv2.CAP_PROP_FPS))
    except AttributeError as err:
        print(f'{err}\nDefault fps value set: 30')
        fps = 30

    # A list of boolean values which represent whether there was somebody from reference dataset detected
    presence_of_reference = [False for i in range(conf.check_every_m_analyses)]

    # Instantiate necessary values
    video_writer, video_name, currently_detected, N_counter = None, "", set(), 1
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            break

        if N_counter == conf.n_th_frame:
            detected_labels = classifier.get_labels(frame)

            if len(detected_labels) != 0:
                presence_of_reference.append(True)
                currently_detected.update(detected_labels)
                print(f'Identities of interest currently present in the recorded clip: {currently_detected}')
            else:
                presence_of_reference.append(False)

            # Remove the oldest detection
            presence_of_reference = presence_of_reference[1:]

            if True in presence_of_reference:
                if video_writer is None:
                    print('Recording started')
                    # Instantiate the video writer
                    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
                    video_name = f'{time()}.avi'
                    video_writer = cv2.VideoWriter(join(conf.VIDEO_OUT, video_name),
                                                   cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width, frame_height))
            elif video_writer is not None:
                print('Recording stopped')
                video_writer.release()
                video_writer = None
                with open(join(conf.VIDEO_OUT, f'{video_name}_identities'), 'a') as f:
                    f.write(",".join(currently_detected))

            N_counter = 1
        else:
            N_counter += 1

        if video_writer is not None:
            video_writer.write(frame)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    if video_writer is not None:
        video_writer.release()
        with open(join(conf.VIDEO_OUT, f'{video_name}_identities'), 'a') as f:
            f.write(",".join(currently_detected))
