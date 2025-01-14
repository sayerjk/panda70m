"""
This script processes video files to analyze human presence using Google's API.

It performs the following tasks:
1. Logs the processing status of each video.
2. Connects to the Google API for video analysis.
3. Uploads videos to the Google API and waits for processing.
4. Analyzes the presence of humans in the video and generates a JSON output.
5. Logs the results of the analysis.

Classes:
    ProcessingStatus(Enum): Enum for processing statuses.
    VideoProcessingLogger: Handles logging of video processing statuses.

Functions:
    connect_to_google_api(): Connects to the Google API using credentials from the environment.
    select_video(video_basename: str): Selects a video file based on its basename.
    await_upload(file_upload): Waits for the video upload to complete.
    analyze_human_presence(video_id: str, logger: VideoProcessingLogger): Analyzes human presence in a video.

Usage:
    Run the script to process all pending videos in the specified directory.
"""

import datetime
import json
import os
import time
import csv
from dotenv import load_dotenv
from IPython.display import Markdown
from typing import List
from google import genai
from google.genai import types
import pathlib
from enum import Enum
import glob
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import subprocess
from pathlib import Path

from prompts import *

# Global variables for file tracking
active_files = set()
active_files_lock = threading.Lock()

ds_base_dir = os.path.abspath('D:\\Datasets\\Video\\Panda-70M\\dataset')
video_dir = os.path.join(ds_base_dir, 'panda70m_hq6m_filtered_humans_v2')
all_mp4_files = list(pathlib.Path(video_dir).rglob('*.mp4'))
# check_list = os.listdir('D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_filtered_humans_v2')
# MODEL_ID = "gemini-2.0-flash-exp"  # @param ["gemini-1.5-flash-8b","gemini-1.5-flash-002","gemini-1.5-pro-002","gemini-2.0-flash-exp"] {"allow-input":true}
# MODEL_ID = 'gemini-1.5-pro-002' # free tier: 2 requests per minute
# TODO - test 2.0 again; 1.5-flash has higher/better rate limits
MODEL_ID = 'gemini-1.5-flash-002'
USER_PROMPT = GRAPHICS_USER_PROMPT_v2  # << set prompt before running >>
# USER_PROMPT = USER_PROMPT

class ProcessingStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class VideoProcessingLogger:
    def __init__(self, base_dir: str, analysis_type: str):
        self.log_dir = os.path.join(ds_base_dir, 'processing_logs')
        self.analysis_type = analysis_type
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Include analysis type in log filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.log_dir, f'processing_status_{analysis_type}_{timestamp}.csv')
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'status', 'timestamp', 'error_message'])
        
        print(f"Logging to: {self.csv_path}")
    
    def get_all_logs(self) -> List[str]:
        """Get all previous log files for this analysis type"""
        return sorted(glob.glob(os.path.join(self.log_dir, f'processing_status_{self.analysis_type}_*.csv')))
    
    def get_pending_videos(self, all_video_ids: List[str]) -> List[str]:
        completed_videos = set()
        
        # Check all previous logs for completed videos
        for log_file in self.get_all_logs():
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['status'] == ProcessingStatus.COMPLETED.value:
                        completed_videos.add(row['video_id'])
        
        return [vid for vid in all_video_ids if vid not in completed_videos]

    def log_status(self, video_id: str, status: ProcessingStatus, error_message: str = None):
        """Log the processing status of a video
        
        Args:
            video_id: The ID of the video being processed
            status: ProcessingStatus enum value
            error_message: Optional error message if status is FAILED
        """
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                video_id,
                status.value,
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                error_message or ''
            ])

class AnalysisConfig:
    def __init__(self, metadata_type: str, force_reprocess: bool = False, max_duration: float = 120.0):
        self.metadata_type = metadata_type
        self.force_reprocess = force_reprocess
        self.max_duration = max_duration # time in seconds
        
    @property
    def output_suffix(self) -> str:
        return f"tc_{self.metadata_type}"

def connect_to_google_api():
    """Create a new API client instance."""
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    return genai.Client(api_key=GOOGLE_API_KEY)


def select_video(video_basename: str):
    video_path = list(filter(lambda file: file.stem == video_basename, all_mp4_files))
    assert len(video_path) == 1, f"Expected 1 video name match, but found {len(video_path)} files."
    video_path = video_path[0]
    return video_path


def await_upload(file_upload, client):
    """Wait for video upload to complete."""
    attempts = 0
    max_attempts = 25  # Add maximum attempts to prevent infinite loops
    
    while file_upload.state == "PROCESSING" and attempts < max_attempts:
        print('Waiting for video to be processed.')
        time.sleep(3)
        file_upload = client.files.get(name=file_upload.name)
        attempts += 1

    if file_upload.state == "FAILED" or attempts >= max_attempts:
        raise ValueError(f"Upload failed or timed out: {file_upload.state}")
    print(f'Video processing complete: ' + file_upload.uri)
    time.sleep(1)  # Add small cooldown after successful upload

log_lock = threading.Lock()

def safe_log_status(logger, video_id, status, error_message=None):
    """Thread-safe wrapper for logging status"""
    with log_lock:
        logger.log_status(video_id, status, error_message)

def parse_timestamp(time_str: str) -> str:
    """Parse and normalize timestamp to MM:SS format.
    Raises ValueError if the timestamp is not in a valid format."""
    # If it's already in MM:SS format, validate and return
    if time_str.count(':') == 1:
        minutes, seconds = map(int, time_str.split(':'))
        if minutes < 0 or seconds < 0 or seconds >= 60:
            raise ValueError(f"Invalid minutes/seconds values in timestamp: {time_str}")
        return f"{minutes:02d}:{seconds:02d}"
        
    # If it's in HH:MM:SS format, convert to MM:SS
    if time_str.count(':') == 2:
        hours, minutes, seconds = map(int, time_str.split(':'))
        total_minutes = hours * 60 + minutes
        if total_minutes < 0 or seconds < 0 or seconds >= 60:
            raise ValueError(f"Invalid hours/minutes/seconds values in timestamp: {time_str}")
        return f"{total_minutes:02d}:{seconds:02d}"
        
    raise ValueError(f"Invalid timestamp format: {time_str}")

def validate_timecode_sequence(timecodes: List[dict], video_id: str) -> None:
    """Validate that timecodes form a proper sequence starting at 0:00."""
    if not timecodes:
        raise ValueError(f"Empty timecode sequence for video {video_id}")
        
    # Check first timecode is 0:00
    if timecodes[0]['time'] != "00:00":
        raise ValueError(f"First timecode must be 00:00, got {timecodes[0]['time']} for video {video_id}")
    
    # Check for consistent 1-second intervals
    for i in range(1, len(timecodes)):
        prev_min, prev_sec = map(int, timecodes[i-1]['time'].split(':'))
        curr_min, curr_sec = map(int, timecodes[i]['time'].split(':'))
        prev_total_secs = prev_min * 60 + prev_sec
        curr_total_secs = curr_min * 60 + curr_sec
        
        if curr_total_secs - prev_total_secs != 1:
            raise ValueError(f"Non-consecutive timecodes at index {i}: {timecodes[i-1]['time']} -> {timecodes[i]['time']} for video {video_id}")

def get_video_duration(video_path: Path) -> float:
    """Get duration of video in seconds using ffmpeg."""
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', 
               '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
        output = subprocess.check_output(cmd).decode().strip()
        return float(output)
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting duration for {video_path}: {e}")
        return 0

def analyze_video_content(video_id: str, logger: VideoProcessingLogger, config: AnalysisConfig):
    file_upload = None
    client = None
    try:
        video_path = select_video(video_id)
        
        # Check video duration
        duration = get_video_duration(video_path)
        if duration >= config.max_duration:
            safe_log_status(logger, video_id, ProcessingStatus.SKIPPED, f"Video duration {duration:.1f}s exceeds limit of {config.max_duration}s")
            return
            
        json_output_path = f'{str(video_path.parent)}\\{video_id}_{config.output_suffix}.json'
        
        if not config.force_reprocess and os.path.exists(json_output_path):
            safe_log_status(logger, video_id, ProcessingStatus.SKIPPED, "Output file already exists")
            return
        
        safe_log_status(logger, video_id, ProcessingStatus.PENDING)
        
        # Create a new client for each video
        client = connect_to_google_api()
        file_upload = client.files.upload(path=video_path)
        
        # Add to active files
        with active_files_lock:
            active_files.add(file_upload.name)
            
        await_upload(file_upload, client)

        set_timecodes = types.FunctionDeclaration(
            name="set_timecodes",
            description="Set the timecodes for the video with associated text",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "text": {"type": "STRING"},
                            },
                            "required": ["time", "text"],
                        }
                    }
                },
                "required": ["timecodes"]
            }
        )

        set_timecodes_with_objects = types.FunctionDeclaration(
            name="set_timecodes_with_objects",
            description="Set the timecodes for the video with associated text and object list",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "text": {"type": "STRING"},
                                "objects": {
                                    "type": "ARRAY",
                                    "items": {"type": "STRING"},
                                },
                            },
                            "required": ["time", "text", "objects"],
                        }
                    }
                },
                "required": ["timecodes"],
            }
        )

        set_timecodes_with_numeric_values = types.FunctionDeclaration(
            name="set_timecodes_with_numeric_values",
            description="Set the timecodes for the video with associated numeric values",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "value": {"type": "NUMBER"},
                            },
                            "required": ["time", "value"],
                        }
                    }
                },
                "required": ["timecodes"],
            }
        )

        set_timecodes_with_descriptions = types.FunctionDeclaration(
            name="set_timecodes_with_descriptions",
            description="Set the timecodes for the video with associated spoken text and visual descriptions",
            parameters={
                "type": "OBJECT",
                "properties": {
                    "timecodes": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "time": {"type": "STRING"},
                                "spoken_text": {"type": "STRING"},
                                "visual_description": {"type": "STRING"},
                            },
                            "required": ["time", "spoken_text", "visual_description"],
                        }
                    }
                },
                "required": ["timecodes"]
            }
        )

        video_tools = types.Tool(
            function_declarations=[set_timecodes, set_timecodes_with_objects, set_timecodes_with_numeric_values],
        )

        def set_timecodes_func(timecodes):
            return [{**t, "text": t["text"].replace("\\'", "'")} for t in timecodes]

        def set_timecodes_with_objects_func(timecodes):
            return [{**t, "text": t["text"].replace("\\'", "'")} for t in timecodes]

        def set_timecodes_with_descriptions_func(timecodes):
            return [{**t, "text": t["spoken_text"].replace("\\'", "'")} for t in timecodes]


        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=file_upload.uri,
                            mime_type=file_upload.mime_type),
                    ]),
                USER_PROMPT,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[video_tools],
                temperature=0,
            )
        )

        time.sleep(2)

        # Add error checking for response
        if not response.candidates:
            raise ValueError("No response candidates received from API")
            
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            raise ValueError("No content in response candidate")
            
        part = candidate.content.parts[0]
        if not hasattr(part, 'function_call'):
            raise ValueError(f"No function call in response. Response content: {str(candidate.content)}")
            
        if not hasattr(part.function_call, 'args'):
            raise ValueError(f"No args in function call. Function call content: {str(part.function_call)}")

        results = part.function_call.args

        # Validate and process timestamps before saving anything
        try:
            # Parse and normalize timestamps to MM:SS format
            for timecode in results['timecodes']:
                timecode['time'] = parse_timestamp(timecode['time'])

            # Sort the list of dictionaries by the 'time' key
            sorted_timecodes = sorted(results['timecodes'], 
                                    key=lambda x: tuple(map(int, x['time'].split(':'))))

            # Validate timecode sequence
            validate_timecode_sequence(sorted_timecodes, video_id)

            # Ensure 'time' key is first in each dictionary
            sorted_timecodes = [{'time': item['time'], **{k: v for k, v in item.items() if k != 'time'}} 
                              for item in sorted_timecodes]

            results['timecodes'] = sorted_timecodes

            # Only save to JSON if all validation passes
            print(f"Processing {video_id}:")  # Add video ID to output for better tracking
            print(results)
            with open(json_output_path, 'w') as json_file:
                json.dump(results, json_file, indent=4)

            print(f"Video analyzer results saved to [{json_output_path}]")
            safe_log_status(logger, video_id, ProcessingStatus.COMPLETED)

        except ValueError as e:
            # Log the specific timestamp processing error and continue
            error_msg = f"Failed to process timestamps in response: {str(e)}"
            safe_log_status(logger, video_id, ProcessingStatus.FAILED, error_msg)
            raise ValueError(error_msg)

    except Exception as e:
        safe_log_status(logger, video_id, ProcessingStatus.FAILED, str(e))
        raise e
    finally:
        if file_upload and client:
            # Remove from active files before deleting
            with active_files_lock:
                active_files.discard(file_upload.name)
            try:
                client.files.delete(name=file_upload.name)
            except Exception as e:
                print(f"Failed to delete file {file_upload.name}: {e}")

def cleanup_uploaded_files(client):
    """Clean up any uploaded files that aren't currently being processed."""
    try:
        with active_files_lock:
            current_active = active_files.copy()
        
        files = client.files.list()
        for file in files:
            if file.name not in current_active:  # Only delete if not in use
                try:
                    client.files.delete(name=file.name)
                    print(f"Cleaned up inactive file: {file.name}")
                except Exception as e:
                    print(f"Failed to delete file {file.name}: {e}")
    except Exception as e:
        print(f"Failed to list files: {e}")

if __name__ == "__main__":
    # Configure the analysis
    config = AnalysisConfig(
        metadata_type="graphics" if "graphics" in USER_PROMPT.lower() else "humans",
        force_reprocess=False
    )
    
    # Clean up any lingering files before starting
    startup_client = connect_to_google_api()
    cleanup_uploaded_files(startup_client)
    del startup_client  # Ensure we don't keep this reference around
    
    logger = VideoProcessingLogger('.', config.metadata_type)
    all_video_ids = [file.stem for file in all_mp4_files]
    pending_videos = logger.get_pending_videos(all_video_ids)
    pending_videos = [v for v in pending_videos if '_filter_gemini_' not in v]

    print(f"Found {len(pending_videos)} videos pending {config.metadata_type} analysis")
    print(f"Force reprocess: {config.force_reprocess}")
    
    # Number of concurrent workers - adjust based on your system and API limits
    max_workers = 16
    batch_size = 50  # Process 50 videos at a time
    
    # Create a partial function with the fixed arguments
    analyze_func = partial(analyze_video_content, logger=logger, config=config)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        try:
            # Process videos in batches
            for i in range(0, len(pending_videos), batch_size):
                batch = pending_videos[i:i + batch_size]
                print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} videos)")
                
                # Submit current batch with rate limiting
                futures = []
                for video_id in batch:
                    futures.append(executor.submit(analyze_func, video_id))
                    # Add delay between submissions to avoid overwhelming API
                    time.sleep(1)
                
                # Wait for current batch to complete before starting next batch
                for future in futures:
                    try:
                        future.result()  # This will raise any exceptions that occurred
                    except Exception as e:
                        print(f'Error in worker thread: {str(e)}')
                        time.sleep(2)
                
                # Clean up after each batch
                cleanup_client = connect_to_google_api()
                cleanup_uploaded_files(cleanup_client)
                del cleanup_client
                
        except KeyboardInterrupt:
            print("\nReceived interrupt, shutting down gracefully...")
            executor.shutdown(wait=False, cancel_futures=True)
            raise