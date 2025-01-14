import json
import pandas as pd
from pathlib import Path

def convert_json_to_csv():
    # Define paths
    json_path = Path("./dataset/Panda70M_HQ6M.json")
    output_csv_path = Path("./dataset/panda70m_hq6m.csv")
    
    # Read JSON file
    print("Reading JSON file...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Convert to DataFrame format matching CSV structure
    print("Converting data format...")
    formatted_data = []
    
    for entry in json_data:
        # Extract video ID from path
        video_id = entry['path'].split('/')[1]  # Gets ID from panda70m_part_XXXX/VIDEO_ID/...
        
        # Convert duration to timestamp format ['start', 'end']
        duration = entry['duration']
        timestamp = [['0:00:00.000', f'0:00:{duration:.3f}']]
        
        formatted_entry = {
            'videoID': video_id,
            'url': f"https://www.youtube.com/watch?v={video_id}",
            'timestamp': str(timestamp),  # Convert to string to match CSV format
            'caption': str(entry.get('cap', [])),  # Convert captions list to string
            'matching_score': str([1.0]),  # Placeholder matching score
            'desirable_filtering': str(['desirable']),  # Assuming all entries are desirable
            'shot_boundary_detection': str([timestamp])  # Using same timestamp for shot boundary
        }
        formatted_data.append(formatted_entry)
    
    # Create DataFrame
    df = pd.DataFrame(formatted_data)
    
    # Save to CSV
    print(f"Saving to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False)
    print("Conversion complete!")
    
    # Print sample for verification
    print("\nSample of converted data:")
    print(df.head(1).to_string())

if __name__ == "__main__":
    convert_json_to_csv() 