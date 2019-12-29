from os.path import join
from time import time

import cv2


class VideoProcessor:
    """
    A class which iterates through the frames of moviepy.editor.VideoFileClip
    and starts saving them to a new video file when the target identity
    is detected.
    """

    def __init__(self, classifier_wrapper, video_dir, display_vid=False):
        """
        :param classifier_wrapper: a class implementing the classification pipeline
        :param video_dir: a path to the directory where the processed videos will be stored
        :param display_vid: a parameter determining whether the video should be displayed
               in a separate window
        """
        self.classifier_wrapper = classifier_wrapper
        self.video_dir = video_dir
        self.display_vid = display_vid

    def process(self, video_clip, nth_frame, m_analyses):
        """
        A function which processes the video as is explained in the class description.
        :param video_clip: moviepy.editor.VideoFileClip - the video to process
        :param nth_frame: int, every nth frame will be analysed
        :param m_analyses: int, a number which defines how long to keep recording
               - defined as a number of analyses since the last positive detection
        :return: None - saves the video segments directly to files
        """

        # A list of boolean values which represent whether there was somebody from reference dataset detected
        presence_of_reference = [False for i in range(m_analyses)]

        video_name, currently_detected, n_counter, recording_t, recording = None, set(), 1, 0, False
        # 1) Iterate through the video frames
        for t, video_frame in video_clip.iter_frames(with_times=True):

            # 2)_Check whether it is time to analyze the frame
            if n_counter == nth_frame:
                # 3) Get labels from the frame
                detected_labels = self.classifier_wrapper.get_labels(video_frame)

                # 4) Store the information whether there was someone detected
                # and update the currently_detected set if necessary
                if len(detected_labels) != 0:
                    presence_of_reference.append(True)
                    if len(detected_labels - currently_detected) != 0:
                        currently_detected.update(detected_labels)
                        print(f'Identities of interest currently present in the recorded clip: {currently_detected}')
                else:
                    presence_of_reference.append(False)

                # 5) Remove the oldest detection
                presence_of_reference = presence_of_reference[1:]

                # 6) Check whether there was there was someone detected in the last m analyses
                # and start recording if it is not already the case
                if True in presence_of_reference:
                    if not recording:
                        print('Recording started')
                        video_name, recording_t, recording = f'{time()}.mp4', t, True

                elif recording:
                    # 7) Stop recording if there were not any detections in the last m analyses
                    self._save_recording(video_clip, video_name, currently_detected, recording_t, t)
                    video_name, recording = None, False

                # 8) Reset the counter
                n_counter = 1
            else:
                n_counter += 1

            if self.display_vid:
                # 9) Display the video frame if the corresponding flag was set
                # OpenCV expects the frame in BGR color format
                video_frame_BGR = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow('frame', video_frame_BGR)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if self.display_vid:
            # When everything done, close the windows
            cv2.destroyAllWindows()

        if recording:
            # 10) If the last frame was reached during recording then save the segment
            self._save_recording(video_clip, video_name, currently_detected, recording_t)

    def _save_recording(self, video, video_name, currently_detected, t_start, t_end=None):
        """
        A function which saves the video and corresponding detections
        :param video: oviepy.editor.VideoFileClip - a video which is currently being processed
        :param video_name: str, a name under which the video is going to be saved
        :param currently_detected: a set of identities detected in the video segment
        :param t_start: numpy.float64, the beginning of the video segment
        :param t_end: numpy.float64, the beginning of the video segment
        :return: None - saves the video segments directly to files
        """
        print('Recording stopped')
        video_clip = video.subclip(t_start, t_end)
        video_clip.write_videofile(join(self.video_dir, video_name), codec='libx265')
        with open(join(self.video_dir, f'{video_name}_identities'), 'a') as f:
            f.write(",".join(currently_detected))
        currently_detected.clear()
