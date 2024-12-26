import os
import glob

full_aggregated_dataset = 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1'

gemini_metadata = glob.glob(f'{full_aggregated_dataset}\\**\\*_tc_*.json', recursive=True)

video_paths = glob.glob(f'{full_aggregated_dataset}\\**\\*.mp4', recursive=True)
# e.g. 'D:\\Datasets\\Video\\Panda-70M\\dataset\\panda70m_hq6m_formatted_humansOnly_v2.1\\00000\\0000000_00000.mp4'
original_panda_videos = [path for path in video_paths if '_human_' not in os.path.basename(path) and '_graphics_' not in os.path.basename(path)]
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
print(f"Total original videos: {len(original_panda_videos)}")
print(f"Videos with human metadata: {len(gem_humans)}")
print(f"Videos with graphics metadata: {len(gem_graphics)}")
print(f"Videos ready for processing (have both metadata files): {len(ready_for_processing)}")
print(f"Already processed subclips: {len(postprocessed_subclips)}")

# Optionally, print some example paths of ready-to-process videos
if ready_for_processing:
    print("\nExample videos ready for processing:")
    for path in ready_for_processing[:3]:  # Show first 3 examples
        print(f"- {path}")