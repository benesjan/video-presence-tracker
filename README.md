# video-presence-tracker

video-presence-tracker is a Python project which cuts out video segments from videos containing
specified identities.
The system uses face detection and recognition implemented in PyTorch.

## Installation

1. Clone the project:
    ```bash
    git clone https://github.com/benesjan/video-presence-tracker.git
    ```

2. Enter the directory and install the dependencies:
    ```bash
    cd video-presence-tracker
    pip install -r requirements.txt
    ```

3. Download the [model weights](https://drive.google.com/open?id=1wJTbgNT11GSLNZT-nsCzXNeu6Pqh5r3y) and set the
MODEL_WEIGHTS constant in the config.py file.

4. Create a reference dataset in the
[standardized format](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder)
(dataset/first_identity/image1.jpg, ...) and set the DATASET constant in the config.py file.
Each image has to contain exactly one face.
If the face is not detected in any of the images corresponding to the identity, new images will have to be provided.

5. Finally, execute the setup.py script:
    ```bash
    python ./setup.py
    ```

## Usage
- To process a specific video, run the following command:
    ```bash
    python ./process_video.py --display-video --video-path path/to/video.mp4
    ```
    If the --display-video flag is present, the video will be displayed in a window as it is processed.
    
- The track_yt_channel.py continually processes new videos uploaded to a YouTube channel specified by the CHANNEL_ID.
    To execute the script run the command bellow:
    ```bash
    python ./track_yt_channel.py --display-video --channel-id CHANNEL_ID --yt-api-key API_KEY
    ```
    To obtain the CHANNEL_ID go to [this site](https://socialnewsify.com/get-channel-id-by-username-youtube/)
    and enter the channel name.
    The get the YouTube API key read the Data API [documentation](https://developers.google.com/youtube/v3/getting-started). 

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
[MIT](https://choosealicense.com/licenses/mit/)