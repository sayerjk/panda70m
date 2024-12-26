"""
This script processes video files by extracting clips where humans are detected, using output from Google Gemini in
video_anaylzer.py and FFmpeg for fast trimming or re-encoding
"""

import os
import json
import glob
import subprocess
from datetime import timedelta

def extract_human_clips(video_path, json_path, output_path=None):
    """
    Extracts subclips from a video where humans are detected, as specified in a JSON file.

    Args:
        video_path (str): Path to the input video file.
        json_path (str): Path to the JSON file with metadata about human detections.
        output_path (str): Directory to save the output video clips. If None, uses the same directory as the input video.
    """
    import os

    if output_path is None:
        output_path = os.path.dirname(video_path)

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    with open(json_path, 'r') as f:
        metadata = json.load(f)

    # Parse timecodes with human detections
    timecodes = metadata["timecodes"]

    # Identify segments where humans are detected
    segments = []
    start = None

    for tc in timecodes:
        time_in_seconds = int(tc["time"].split(":")[1]) + int(tc["time"].split(":")[0]) * 60
        if tc["value"] > 0:
            if start is None:
                start = time_in_seconds
        else:
            if start is not None:
                segments.append((start, time_in_seconds))
                start = None

    # If a segment is still open, close it at the end of the video
    if start is not None:
        video_duration = get_video_duration(video_path)
        segments.append((start, video_duration))

    # Extract and save clips using FFmpeg
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    for idx, (start, end) in enumerate(segments):
        start_time = str(timedelta(seconds=start))
        duration = end - start
        idx_suffix = idx + 1
        output_file = os.path.join(output_path, f"{base_name}_human_{idx_suffix:05d}.mp4")

        # FFmpeg command to extract the clip
        command = [
            "ffmpeg",
            "-ss", start_time,
            "-i", video_path,
            "-t", str(duration),
            "-an",  # Disable audio (explicitly since input has no audio)
            "-c:v", "copy",  # Fastest, no re-encoding
            output_file
        ]

        # Print and run the FFmpeg command for debugging
        print("Running command:", " ".join(command))
        subprocess.run(command, check=True)

    print(f"Clips have been saved to {output_path}")

def get_video_duration(video_path):
    """
    Get the duration of a video file in seconds.

    Args:
        video_path (str): Path to the video file.

    Returns:
        int: Duration of the video in seconds.
    """
    import ffmpeg
    probe = ffmpeg.probe(video_path)
    duration = float(probe['format']['duration'])
    return int(duration)

if __name__ == "__main__":

    video_storage = 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1'
    gemini_metadata = glob.glob(f'{video_storage}\\**\\*_tc_humans.json', recursive=True)
    video_paths = glob.glob(f'{video_storage}\\**\\*.mp4', recursive=True)

    video_paths = [path for path in video_paths if
                   '_clip_' not in os.path.basename(path) and
                   os.path.exists(path.replace('.mp4', '_tc_humans.json'))]

    from tqdm import tqdm

    for video_path in tqdm(video_paths, desc="Processing videos"):
        json_path = video_path.replace('.mp4', '_tc_humans.json')
        extract_human_clips(video_path, json_path)

    # video = "D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1\\00000\\0000008_00000.mp4"
    # metadata = "D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1\\00000\\0000008_00000_tc_humans.json"
    #
    # extract_human_clips(video, metadata)
