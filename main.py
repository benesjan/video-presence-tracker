from argparse import ArgumentParser

import cv2
from pafy import pafy

from classifierwrapper import ClassifierWrapper
from config import Config
from utils import load_model, load_pickle


def start_recording():
    print("Start recording")


def stop_recording():
    print("Stop recording")


if __name__ == '__main__':
    parser = ArgumentParser(description='A bot saving video clips from YouTube video stream'
                                        'containing people whose faces are in the dataset.')
    parser.add_argument("--url", help="URL of the video stream.", required=True, type=str)
    args = parser.parse_args()

    # Load configuration and model
    conf = Config()

    # Load reference features and labels.
    ref_labels, ref_features = load_pickle(conf.REPRESENTATIONS)

    classifier = ClassifierWrapper(load_model(conf), ref_labels, ref_features, conf.DEVICE)

    # Get the video stream and pass it to OpenCV
    video = pafy.new(args.url)
    stream = video.getbest(preftype="mp4")
    # stream = video.allstreams[4]
    cap = cv2.VideoCapture()
    cap.open(stream.url)

    # A list of boolean values which represent whether there was somebody from reference dataset detected
    presence_of_reference = [False for i in range(conf.check_every_m_analyses)]

    N_counter = 1
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()

        if N_counter == conf.n_th_frame:
            detected_labels = classifier.get_labels(frame)

            presence_of_reference.append(len(detected_labels) != 0)
            # Remove the oldest detection
            presence_of_reference = presence_of_reference[1:]

            if True in presence_of_reference:
                start_recording()
            else:
                stop_recording()

            N_counter = 1
        else:
            N_counter += 1

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
