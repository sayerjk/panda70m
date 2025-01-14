"""
This script assesses video files by analyzing their metadata and categorizing them into original videos and post-processed subclips.
It identifies videos that are ready for further processing based on the presence of specific metadata files.

The script performs the following tasks:
1. Lists all video files and their associated metadata in the specified dataset directory.
2. Categorizes videos into original videos and post-processed subclips.
3. Identifies original videos that have both human and graphics metadata files.
4. Prints a summary of the analysis results, including counts of various categories and example paths of videos ready for processing.

Usage:
- Update the `full_aggregated_dataset` variable to point to the desired dataset directory.
- Run the script to perform the analysis and print the results.
"""

import os
import glob

# full_aggregated_dataset = 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1'
full_aggregated_dataset = 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_filtered_humans_v2'


gemini_metadata = glob.glob(f'{full_aggregated_dataset}\\**\\*_tc_*.json', recursive=True)

video_paths = glob.glob(f'{full_aggregated_dataset}\\**\\*.mp4', recursive=True)
# e.g. 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1\\00000\\0000000_00000.mp4'
# original_panda_videos = [path for path in video_paths if '_human_' not in os.path.basename(path) and '_graphics_' not in os.path.basename(path)]
original_panda_videos = [path for path in video_paths if '_filter_gemini_' not in os.path.basename(path)]
# e.g. 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1\\00000\\0000001_00000_human_00001.mp4'
postprocessed_subclips = [path for path in video_paths if path not in original_panda_videos]

print(f"Original videos: {len(original_panda_videos)}")

gem_humans = [path for path in gemini_metadata if '_tc_humans.json' in os.path.basename(path)]
gem_graphics = [path for path in gemini_metadata if '_tc_graphics.json' in os.path.basename(path)]

# Find original videos that have both metadata files
ready_for_processing = []
for video_path in original_panda_videos:
    base_path = video_path[:-4]  # Remove .mp4 extension
    humans_json = f"{base_path}_tc_humans.json"
    graphics_json = f"{base_path}_tc_graphics.json"
    
    if os.path.exists(humans_json) and os.path.exists(graphics_json):
        ready_for_processing.append(video_path)


print(f"\nAnalysis Results:")
print(f"Total downloaded videos (Panda70M_HQ6M): {len(original_panda_videos)}")
print(f"Videos with human metadata: {len(gem_humans)} ({len(gem_humans) / len(original_panda_videos) * 100:.2f}%)")
print(f"Videos with graphics metadata: {len(gem_graphics)} ({len(gem_graphics) / len(original_panda_videos) * 100:.2f}%)")
print(f"Videos ready for processing (have both metadata files): {len(ready_for_processing)} ({len(ready_for_processing) / len(original_panda_videos) * 100:.2f}%)")
print(f"Already processed subclips (had `human` + `graphics` metadata): {len(postprocessed_subclips)} ({len(postprocessed_subclips) / len(video_paths) * 100:.2f}%)")


# Optionally, print some example paths of ready-to-process videos
if ready_for_processing:
    print("\nExample videos ready for processing:")
    for path in ready_for_processing[:3]:  # Show first 3 examples
        print(f"- {path}")