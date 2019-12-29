from os.path import join
from time import time

import cv2


class VideoProcessor:
    def __init__(self, ref_labels, ref_features, classifier, video_dir, display_vid=False):
        self.ref_labels = ref_labels
        self.ref_features = ref_features
        self.classifier = classifier
        self.video_dir = video_dir
        self.display_vid = display_vid

    def process(self, video_clip, nth_frame, record_if_in_m_anal):
        # A list of boolean values which represent whether there was somebody from reference dataset detected
        presence_of_reference = [False for i in range(record_if_in_m_anal)]

        video_name, currently_detected, N_counter, recording_t, recording = None, set(), 1, 0, False
        for t, video_frame in video_clip.iter_frames(with_times=True):

            if N_counter == nth_frame:
                detected_labels = self.classifier.get_labels(video_frame)

                if len(detected_labels) != 0:
                    presence_of_reference.append(True)
                    if len(detected_labels - currently_detected) != 0:
                        currently_detected.update(detected_labels)
                        print(f'Identities of interest currently present in the recorded clip: {currently_detected}')
                else:
                    presence_of_reference.append(False)

                # Remove the oldest detection
                presence_of_reference = presence_of_reference[1:]

                if True in presence_of_reference:
                    if not recording:
                        print('Recording started')
                        video_name, recording_t, recording = f'{time()}.mp4', t, True

                elif recording:
                    self._save_recording(video_clip, video_name, currently_detected, recording_t, t)
                    video_name, recording = None, False

                N_counter = 1
            else:
                N_counter += 1

            if self.display_vid:
                # OpenCV expects frame in BGR color format
                video_frame_BGR = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                # Display the resulting frame
                cv2.imshow('frame', video_frame_BGR)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.display_vid:
            # When everything done, close the windows
            cv2.destroyAllWindows()

        if recording:
            self._save_recording(video_clip, video_name, currently_detected, recording_t)

    def _save_recording(self, video, video_name, currently_detected, t_start, t_end=None):
        print('Recording stopped')
        video_clip = video.subclip(t_start, t_end)
        video_clip.write_videofile(join(self.video_dir, video_name), codec='libx265')
        with open(join(self.video_dir, f'{video_name}_identities'), 'a') as f:
            f.write(",".join(currently_detected))
        currently_detected.clear()
