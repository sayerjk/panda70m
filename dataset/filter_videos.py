"""
This script processes video files by extracting clips based on multiple filtering conditions,
using output from Google Gemini in video_analyzer.py and FFmpeg for fast trimming or re-encoding
"""

import os
import json
import glob
import subprocess
from datetime import timedelta
from typing import List, Dict, Tuple, Optional, Callable

class FilterCondition:
    def __init__(self, json_suffix: str, evaluation_func: Callable[[float], bool]):
        """
        Args:
            json_suffix (str): Suffix for the JSON metadata file (e.g., '_tc_humans', '_tc_graphics')
            evaluation_func (Callable[[float], bool]): Function that evaluates if a timecode value meets the condition
        """
        self.json_suffix = json_suffix
        self.evaluation_func = evaluation_func

def load_and_validate_metadata(video_path: str, filter_conditions: List[FilterCondition]) -> List[Dict]:
    """
    Loads and validates metadata files for all filter conditions.
    
    Args:
        video_path (str): Path to the video file
        filter_conditions (List[FilterCondition]): List of filter conditions to check
        
    Returns:
        List[Dict]: List of loaded metadata dictionaries
    """
    base_path = os.path.splitext(video_path)[0]
    metadata_list = []
    
    for condition in filter_conditions:
        json_path = f"{base_path}{condition.json_suffix}.json"
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Missing metadata file: {json_path}")
            
        with open(json_path, 'r') as f:
            metadata_list.append(json.load(f))
            
    return metadata_list

def find_valid_segments(metadata_list: List[Dict], 
                       filter_conditions: List[FilterCondition]) -> List[Tuple[int, int]]:
    """
    Identifies video segments that meet all filtering conditions.
    
    Args:
        metadata_list (List[Dict]): List of metadata dictionaries
        filter_conditions (List[FilterCondition]): List of filter conditions
        
    Returns:
        List[Tuple[int, int]]: List of (start, end) time segments in seconds
    """
    # Initialize timeline for each condition
    timelines = []
    for metadata, condition in zip(metadata_list, filter_conditions):
        timeline = {}
        for tc in metadata["timecodes"]:
            time_in_seconds = int(tc["time"].split(":")[1]) + int(tc["time"].split(":")[0]) * 60
            timeline[time_in_seconds] = condition.evaluation_func(tc["value"])
        timelines.append(timeline)
    
    # Find segments where all conditions are met
    segments = []
    start = None
    max_time = max(max(timeline.keys()) for timeline in timelines)
    
    for t in range(max_time + 1):
        # Check if all conditions are met at this time
        all_conditions_met = all(
            timeline.get(t, False) for timeline in timelines
        )
        
        if all_conditions_met:
            if start is None:
                start = t
        else:
            if start is not None:
                segments.append((start, t))
                start = None
                
    # Close final segment if needed
    if start is not None:
        segments.append((start, max_time))
        
    return segments

def check_existing_clips(video_path: str, output_path: str, clip_suffix: str) -> bool:
    """
    Check if clips already exist for the given video.
    
    Args:
        video_path (str): Path to the input video
        output_path (str): Directory to check for existing clips
        clip_suffix (str): Suffix used in output filenames
        
    Returns:
        bool: True if clips exist, False otherwise
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    existing_clips = glob.glob(os.path.join(output_path, f"{base_name}_{clip_suffix}_*.mp4"))
    return len(existing_clips) > 0

def extract_filtered_clips(video_path: str, 
                         filter_conditions: List[FilterCondition], 
                         output_path: Optional[str] = None,
                         clip_suffix: str = "filter_gemini_human+graphics",
                         overwrite: bool = False) -> None:
    """
    Extracts video clips that meet all specified filtering conditions.
    
    Args:
        video_path (str): Path to the input video
        filter_conditions (List[FilterCondition]): List of filter conditions
        output_path (Optional[str]): Output directory (defaults to input video directory)
        clip_suffix (str): Suffix to add to output clip filenames
        overwrite (bool): If False, skip processing if clips already exist
    """
    if output_path is None:
        output_path = os.path.dirname(video_path)
    os.makedirs(output_path, exist_ok=True)
    
    # Check for existing clips if not overwriting
    if not overwrite and check_existing_clips(video_path, output_path, clip_suffix):
        print(f"Skipping {video_path} - output clips already exist")
        return
    
    # Load and validate all metadata
    metadata_list = load_and_validate_metadata(video_path, filter_conditions)
    
    # Find valid segments
    segments = find_valid_segments(metadata_list, filter_conditions)
    
    # Extract clips
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for idx, (start, end) in enumerate(segments, 1):
        start_time = str(timedelta(seconds=start))
        duration = end - start
        output_file = os.path.join(output_path, f"{base_name}_{clip_suffix}_{idx:05d}.mp4")
        
        command = [
            "ffmpeg",
            "-ss", start_time,
            "-i", video_path,
            "-t", str(duration),
            "-an",
            "-c:v", "copy",
            output_file
        ]
        
        print("Running command:", " ".join(command))
        subprocess.run(command, check=True)
        
    print(f"Clips have been saved to {output_path}")

if __name__ == "__main__":
    video_storage = 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_filtered_humans_v2'
    
    # Define filtering conditions
    filter_conditions = [
        FilterCondition(
            json_suffix='_tc_humans',
            evaluation_func=lambda x: x > 0  # At least one human detected
        ),
        FilterCondition(
            json_suffix='_tc_graphics',
            evaluation_func=lambda x: x == 0  # No graphics detected
        )
    ]
    

    
    # Find videos that have all required metadata files
    video_paths = glob.glob(f'{video_storage}\\**\\*.mp4', recursive=True)
    valid_videos = []
    
    for video_path in video_paths:
        if '_filter_gemini_human+graphics_' in video_path:  # Skip already processed videos
            continue
            
        base_path = os.path.splitext(video_path)[0]
        if all(os.path.exists(f"{base_path}{condition.json_suffix}.json") 
               for condition in filter_conditions):
            valid_videos.append(video_path)
    
    # Process videos
    from tqdm import tqdm
    # Configuration
    OVERWRITE_EXISTING = False  # Set to True to reprocess existing clips

    for video_path in tqdm(valid_videos, desc="Processing videos"):
        try:
            extract_filtered_clips(
                video_path, 
                filter_conditions,
                overwrite=OVERWRITE_EXISTING
            )
        except Exception as e:
            print(f"Error processing {video_path}: {str(e)}")
