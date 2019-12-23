from argparse import ArgumentParser

import cv2
import numpy as np
from mtcnn import detect_faces
from pafy import pafy
from scipy.spatial.distance import cdist

from config import Config
from utils import load_model, frontalize_face, process_face, predict, load_pickle

if __name__ == '__main__':
    parser = ArgumentParser(description='A bot saving video clips from YouTube video stream'
                                        'containing people whose faces are in the dataset.')
    parser.add_argument("--url", help="URL of the video stream.", required=True, type=str)
    args = parser.parse_args()

    # Load configuration and model
    conf = Config()
    model = load_model(conf)

    # Get the video stream and pass it to OpenCV
    video = pafy.new(args.url)
    stream = video.getbest(preftype="mp4")
    # stream = video.allstreams[4]
    cap = cv2.VideoCapture()
    cap.open(stream.url)

    ref_labels, ref_features = load_pickle(conf.REPRESENTATIONS)

    N_counter, bboxes = 1, []
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if N_counter == conf.n_th_frame:
            try:
                bboxes, landmarks = detect_faces(frame)
            except Exception as err:
                print(f'Exception: {err}')
                continue

            if len(bboxes) != 0:
                bboxes = bboxes.astype('int32')
                faces = []
                for box_landmarks in landmarks:
                    face_img = frontalize_face(frame, box_landmarks)
                    face_img = process_face(face_img)
                    faces.append(face_img)
                features = predict(model, faces)

                # Classification
                dists = cdist(features, ref_features, metric='cosine')
                decisions = dists < 0.65
                presence = np.sum(decisions, axis=0) >= 1
                label_indices = np.where(presence)[0]
                for label_i in label_indices:
                    print(ref_labels[label_i])

            N_counter = 1
        else:
            N_counter += 1

        for bbox in bboxes:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
