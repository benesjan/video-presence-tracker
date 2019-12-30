# video-presence-tracker

A Python project which cuts out video segments from videos containing
specified identities.
The system uses face detection and recognition implemented in PyTorch.

> This project was created as a part of [gitcoin contest](https://gitcoin.co/issue/ArweaveTeam/Bounties/20/3827)
> organized by [Arweave](https://www.arweave.org/).

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

3. Create a dataset with identities to track.
    The data has to be in the standard format:
    * dataset
        * name1
            * image1.jpg
            * image2.jpg
            * image3.jpg
        * name2
            * image1.jpg
            * image2.jpg
            * ...

    Each image has to contain exactly one face.
    If the face is not detected in any of the images corresponding to the identity, new images will have to be provided.

4. Set the DATASET constant in the config.py file aiming at the dataset root folder.

5. Finally, execute the setup.py script:
    ```bash
    python ./setup.py
    ```
   This command downloads the model weights and computes representative feature vectors from the
   provided dataset.

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