"""
This script filters the Panda70M dataset to identify and extract clips containing human content.

The script processes the filtered CSV (created from HQ6M JSON reference) and performs:
1. Caption analysis using predefined human-related terms
2. Parallel processing of large datasets using ProcessPoolExecutor
3. Preservation of all metadata (timestamps, scores, filtering flags)
4. Statistical reporting of filtering results

Input: Takes the filtered Panda70M-HQ6M CSV (panda70m_hq6m_filtered.csv)
Output: Produces a new CSV (panda70m_hq6m_filtered_humans_v2.csv) containing only
        clips where human presence is detected in captions.

The filtering uses a comprehensive dictionary of human-related terms covering:
- Person references (person, people, human, etc.)
- Gender terms (man, woman, boy, girl, etc.)
- Group terms (crowd, audience, etc.)
- Roles (worker, student, etc.)
- Body parts (face, hand, etc.)
- Actions (walking, sitting, etc.)
"""

import ast
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Define human-related terms for filtering (same as before)
HUMAN_TERMS = {
    'person': ['person', 'people', 'human', 'individual', 'somebody', 'someone'],
    'gender': ['man', 'woman', 'boy', 'girl', 'male', 'female', 'lady', 'gentleman', 'guy'],
    'groups': ['crowd', 'group', 'audience', 'couple', 'family', 'team'],
    'roles': ['worker', 'student', 'teacher', 'doctor', 'player', 'dancer', 'singer', 'actor', 'actress'],
    'body_parts': ['face', 'hand', 'arm', 'leg', 'body', 'hair', 'head', 'finger', 'eye', 'mouth', 'shoulder'],
    'actions': ['walking', 'sitting', 'standing', 'talking', 'dancing', 'running', 'speaking']
}
ALL_HUMAN_TERMS = [term for terms in HUMAN_TERMS.values() for term in terms]

def has_human_terms(caption: str) -> bool:
    """Check if caption contains any human-related terms."""
    return any(term in caption.lower() for term in ALL_HUMAN_TERMS)

def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Process a chunk of the DataFrame."""
    results = []
    for _, row in chunk.iterrows():
        try:
            # Parse all list-like columns
            timestamps = ast.literal_eval(row['timestamp']) if isinstance(row['timestamp'], str) else row['timestamp']
            captions = ast.literal_eval(row['caption']) if isinstance(row['caption'], str) else row['caption']
            matching_scores = ast.literal_eval(row['matching_score']) if isinstance(row['matching_score'], str) else row['matching_score']
            desirable_filtering = ast.literal_eval(row['desirable_filtering']) if isinstance(row['desirable_filtering'], str) else row['desirable_filtering']
            shot_boundaries = ast.literal_eval(row['shot_boundary_detection']) if isinstance(row['shot_boundary_detection'], str) else row['shot_boundary_detection']

            # Create list of indices where captions contain human terms
            human_indices = [i for i, cap in enumerate(captions) if has_human_terms(cap)]

            # If no human clips found, skip this row
            if not human_indices:
                continue

            # Filter all lists to keep only clips with humans
            row = row.copy()
            row['timestamp'] = str([timestamps[i] for i in human_indices])
            row['caption'] = str([captions[i] for i in human_indices])
            row['matching_score'] = str([matching_scores[i] for i in human_indices])
            row['desirable_filtering'] = str([desirable_filtering[i] for i in human_indices])
            row['shot_boundary_detection'] = str([shot_boundaries[i] for i in human_indices])
            
            results.append(row)
        except Exception as e:
            print(f"Error processing row {row['videoID']}: {str(e)}")
            continue
    
    return pd.DataFrame(results) if results else pd.DataFrame(columns=chunk.columns)

def main():
    # Input/Output paths
    input_path = Path("./dataset/panda70m_hq6m_filtered.csv")
    output_path = Path("./dataset/panda70m_hq6m_filtered_humans_v2.csv")
    
    # Read CSV
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Process in chunks
    chunk_size = 1000  # Adjust this based on your available memory
    num_processes = 18  # Adjust based on your CPU cores
    
    chunks = np.array_split(df, max(1, len(df) // chunk_size))
    processed_chunks = []
    
    print(f"Processing {len(chunks)} chunks with {num_processes} processes...")
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        for i, result_df in enumerate(executor.map(process_chunk, chunks)):
            if not result_df.empty:
                processed_chunks.append(result_df)
            print(f"Processed chunk {i+1}/{len(chunks)}")
    
    # Combine results
    print("Combining results...")
    filtered_df = pd.concat(processed_chunks, ignore_index=True)
    
    # Print statistics
    total_clips_before = sum(len(ast.literal_eval(caps)) for caps in df['caption'])
    total_clips_after = sum(len(ast.literal_eval(caps)) for caps in filtered_df['caption'])
    print(f"\nFiltering statistics:")
    print(f"Videos before: {len(df)}")
    print(f"Videos after: {len(filtered_df)}")
    print(f"Total clips before: {total_clips_before}")
    print(f"Total clips after: {total_clips_after}")
    
    # Save filtered data
    print(f"Saving to {output_path}...")
    filtered_df.to_csv(output_path, index=False)
    print("Done!")

if __name__ == "__main__":
    main() 