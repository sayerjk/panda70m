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


all_mp4_files = list(pathlib.Path('.\\panda70m_hq6m_formatted_humansOnly_v2.1').rglob('*.mp4'))
MODEL_ID = "gemini-2.0-flash-exp"  # @param ["gemini-1.5-flash-8b","gemini-1.5-flash-002","gemini-1.5-pro-002","gemini-2.0-flash-exp"] {"allow-input":true}


class ProcessingStatus(Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class VideoProcessingLogger:
    def __init__(self, base_dir: str):
        self.log_dir = os.path.join(base_dir, 'processing_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a new log file for this run with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.log_dir, f'processing_status_{timestamp}.csv')
        
        # Initialize new CSV file
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['video_id', 'status', 'timestamp', 'error_message'])
        
        print(f"Logging to: {self.csv_path}")
    
    def get_all_logs(self) -> List[str]:
        """Get all previous log files"""
        return sorted(glob.glob(os.path.join(self.log_dir, 'processing_status_*.csv')))
    
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

all_mp4_files = list(pathlib.Path('.\\panda70m_hq6m_formatted_humansOnly_v2.1').rglob('*.mp4'))
MODEL_ID = "gemini-2.0-flash-exp"  # @param ["gemini-1.5-flash-8b","gemini-1.5-flash-002","gemini-1.5-pro-002","gemini-2.0-flash-exp"] {"allow-input":true}

def connect_to_google_api():
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    global client
    client = genai.Client(api_key=GOOGLE_API_KEY)
    return client


def select_video(video_basename: str):
    video_path = list(filter(lambda file: file.stem == video_basename, all_mp4_files))
    assert len(video_path) == 1, f"Expected 1 video name match, but found {len(video_path)} files."
    video_path = video_path[0]
    return video_path


def await_upload(file_upload):
    # Prepare the file to be uploaded
    while file_upload.state == "PROCESSING":
        print('Waiting for video to be processed.')
        time.sleep(10)
        file_upload = client.files.get(name=file_upload.name)

    if file_upload.state == "FAILED":
        raise ValueError(file_upload.state)
    print(f'Video processing complete: ' + file_upload.uri)


def analyze_human_presence(video_id: str, logger: VideoProcessingLogger):
    try:
        video_path = select_video(video_id)
        json_output_path = f'{str(video_path.parent)}\\{video_id}_tc_humans.json'
        
        if os.path.exists(json_output_path):
            logger.log_status(video_id, ProcessingStatus.SKIPPED, "Output file already exists")
            return
        
        logger.log_status(video_id, ProcessingStatus.PENDING)
        
        connect_to_google_api()
        file_upload = client.files.upload(path=video_path)
        await_upload(file_upload)

        SYSTEM_PROMPT = "When given a video and a query, call the relevant function only once with the appropriate timecodes and text for the video"
        # prompt for checking human presence
        USER_PROMPT = 'Generate chart data for this video based on the following instructions: for each scene, count the number of people visible. Call set_timecodes_with_numeric_values once with the list of data values and timecodes.'

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
        results = response.candidates[0].content.parts[0].function_call.args

        # Sort the list of dictionaries by the 'time' key
        sorted_timecodes = sorted(results['timecodes'], key=lambda x: datetime.datetime.strptime(x['time'], '%H:%M'))

        # Ensure 'time' key is first in each dictionary
        sorted_timecodes = [{'time': item['time'], **{k: v for k, v in item.items() if k != 'time'}} for item in sorted_timecodes]

        results['timecodes'] = sorted_timecodes

        print(results)

        with open(json_output_path, 'w') as json_file:
            json.dump(results, json_file, indent=4)

        print(f"Video analyzer results saved to [{json_output_path}]")
        
        logger.log_status(video_id, ProcessingStatus.COMPLETED)
        
    except Exception as e:
        logger.log_status(video_id, ProcessingStatus.FAILED, str(e))
        raise e

if __name__ == "__main__":
    logger = VideoProcessingLogger('.')
    all_video_ids = [file.stem for file in all_mp4_files]
    pending_videos = logger.get_pending_videos(all_video_ids)
    
    print(f"Found {len(pending_videos)} videos pending processing")
    
    for video_id in pending_videos:
        print(f'{video_id} starting...')
        try:
            analyze_human_presence(video_id, logger)
        except Exception as e:
            print(f'Error analyzing video {video_id}: {str(e)}')