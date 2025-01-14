# import json
# import pandas as pd
# from pathlib import Path
#
# def convert_json_to_csv():
#     # Define paths
#     json_path = Path("../dataset/Panda70M_HQ6M.json")
#     original_csv_path = Path("../dataset/panda70m_training_full.csv")
#     output_csv_path = Path("../dataset/panda70m_hq6m_filtered.csv")
#
#     # Read JSON file
#     print("Reading JSON file...")
#     with open(json_path, 'r', encoding='utf-8') as f:
#         json_data = json.load(f)
#
#     # Extract video IDs from JSON data
#     json_video_ids = {entry['path'].split('/')[1] for entry in json_data}
#
#     # Read original CSV file
#     print("Reading original CSV file...")
#     original_df = pd.read_csv(original_csv_path)
#
#     # Filter rows based on video IDs in JSON
#     print("Filtering data...")
#     filtered_df = original_df[original_df['videoID'].isin(json_video_ids)]
#
#     # Save filtered data to new CSV
#     print(f"Saving to {output_csv_path}...")
#     filtered_df.to_csv(output_csv_path, index=False)
#     print("Conversion complete!")
#
#     # Print sample for verification
#     print("\nSample of filtered data:")
#     print(filtered_df.head(1).to_string())
#
# if __name__ == "__main__":
#     convert_json_to_csv()


import json
import pandas as pd
import dask.dataframe as dd
from pathlib import Path
import ast

def convert_json_to_csv():
    # Define paths
    json_path = Path("../dataset/Panda70M_HQ6M.json")
    original_csv_path = Path("../dataset/panda70m_training_full.csv")
    output_csv_path = Path("../dataset/panda70m_hq6m_filtered.csv")

    # Read JSON file
    print("Reading JSON file...")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Extract video IDs from JSON data
    json_video_ids = {entry['path'].split('/')[1] for entry in json_data}

    # Read original CSV file using Dask
    print("Reading original CSV file...")
    original_df = dd.read_csv(original_csv_path)

    # Filter rows based on video IDs in JSON
    print("Filtering data...")
    filtered_df = original_df[original_df['videoID'].isin(json_video_ids)]

    # Define a function to convert string representations of lists back to actual lists
    def convert_to_list(value):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value

    # Apply the conversion function to the relevant columns
    columns_to_convert = ['caption', 'desirable_filtering', 'matching_score', 'shot_boundary_detection', 'timestamp']
    for column in columns_to_convert:
        filtered_df[column] = filtered_df[column].apply(convert_to_list, meta=('x', 'object'))

    # Save filtered data to new CSV
    print(f"Saving to {output_csv_path}...")
    filtered_df.to_csv(output_csv_path, index=False, single_file=True)
    print("Conversion complete!")

    # Print sample for verification
    print("\nSample of filtered data:")
    print(filtered_df.head(1).compute().to_string())

if __name__ == "__main__":
    convert_json_to_csv()