import os
import json
import yaml
import logging
import pandas as pd

from tqdm import tqdm
from datetime import datetime


def load_yaml(file_path):
    with open(file_path) as f:
        config = yaml.safe_load(f)
    return config


def load_json(file_path):
    with open(file_path, encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def save_json(data, file_path, overwrite=False):
    """Write data to a JSON file with an option to overwrite or merge with existing content."""
    
    if overwrite or not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
        return
    
    with open(file_path, 'r', encoding='utf-8') as file:
        try:
            existing_data = json.load(file)
        except json.JSONDecodeError:
            existing_data = {}  # Start fresh if the file is empty or corrupt

    if isinstance(existing_data, dict) and isinstance(data, dict):
        for key, value in data.items():
            if key in existing_data and isinstance(existing_data[key], list) and isinstance(value, list):
                existing_data[key].extend(value)
            else:
                existing_data[key] = value
    else:
        raise ValueError("Both existing data and new data must be dictionaries.")

    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(existing_data, file, indent=4, ensure_ascii=False)


def combine_json_files(input_files: list, output_file: str):
    for file in input_files:
        print(f"Processing {file}...")
        json_data = load_json(file)
        save_json(json_data, output_file)
        print(f"Combined data saved to {output_file}")


def combine_json_files_efficient(input_files: list, output_file: str):
    combined_data = {}
    for file in input_files:
        json_data = load_json(file)
        print(f"Processing {file}...")
        for key, value in json_data.items():
            if key in combined_data:
                combined_data[key].extend(value)
            else:
                combined_data[key] = list(value)  # creates a shallow copy of value, given that it's a list
    save_json(combined_data, output_file)
    print(f"Combined data saved to {output_file}")


def configure_logging(script_name, log_level=logging.INFO):
    log_file_path = './logs'
    os.makedirs(log_file_path, exist_ok=True)
    
    log_filename = f"{log_file_path}/{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w"
    )

    # Reduce logging level for API-related libraries
    logging.getLogger("openai").setLevel(logging.WARNING)  # Suppresses OpenAI logs below WARNING
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Suppresses HTTPX logs below WARNING


def append_to_txt_file(data, file_path):
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(data)


def delete_latest_narrative(file_path: str, profiles: list):
    """Deletes the latest narrative for the given profile_id in the specified JSON file."""
    
    data = load_json(file_path)
    for profile_id in profiles:
        # Ensure the profile_id exists in the data
        if profile_id not in data:
            print(f"Profile ID {profile_id} not found.")
            return
        
        # Check if there are narratives to delete
        if not data[profile_id]:
            print(f"No narratives found for Profile ID {profile_id}.")
            return
        
        # Remove the latest narrative (assuming it's stored as a list)
        data[profile_id].pop()

        save_json(data, file_path, overwrite=True)
        
        print(f"Latest narrative for Profile ID {profile_id} deleted successfully.")


def trim_narratives(file_path, k):
    """Keeps only the first k narratives for each profile in a JSON file."""
    
    data = load_json(file_path)

    if not isinstance(data, dict):
        raise ValueError("Expected JSON data to be a dictionary.")

    for profile_id, narratives in data.items():
        if isinstance(narratives, list):
            data[profile_id] = narratives[:k]  # Keep only the first k narratives

    save_json(data, file_path, overwrite=True)


def find_identical_narratives(file_path):
    """Finds profile IDs that have identical sets of narratives in a JSON file."""
    
    data = load_json(file_path)
    
    if not isinstance(data, dict):
        raise ValueError("Expected JSON data to be a dictionary.")

    reverse_map = {}  # Maps tuple of narratives to profile IDs

    for profile_id, narratives in data.items():
        if isinstance(narratives, list):
            narratives_tuple = tuple(sorted(narratives))  # Sort to avoid order differences
            if narratives_tuple in reverse_map:
                reverse_map[narratives_tuple].append(profile_id)
            else:
                reverse_map[narratives_tuple] = [profile_id]

    # Filter out groups with only one profile (no duplicates)
    identical_profiles = {tuple(profiles) for profiles in reverse_map.values() if len(profiles) > 1}

    return identical_profiles

def load_json_narratives_to_df(narr_dir, narr_files):
    df_list = [
        format_narrative_df(pd.read_json(os.path.join(narr_dir,file)), file)
        for file in tqdm(narr_files, desc="Loading narrative file")
    ]
    narr_df = pd.concat(df_list, ignore_index=True)

    return narr_df

def format_narrative_df(df, file_name):
    print(f"Formatting: {file_name}")
    df_formatted = df.melt(var_name='profile_id', value_name='text')
    df_formatted['model'] = file_name.split(sep='_')[0]
    df_formatted['temperature'] = float(file_name.split(sep='_')[1])
    df_formatted['scenario'] = file_name.split(sep='_')[2].split(sep='.')[0]
    df_formatted['sample'] = df_formatted.groupby('profile_id').cumcount() + 1

    return df_formatted

def delete_flagged_row_from_json(
        flagged_rows,
        normalized_df,
        base_path='data/narratives/',
        overwrite=False
    ):
    # Loop through each flagged row to process them
    for idx in flagged_rows:
        row = normalized_df.iloc[idx]  # Get the specific row
        
        # Reconstruct the JSON filename based on columns in the row
        model = row['model']
        temperature = row['temperature']
        scenario = row['scenario']
        sample = row['sample']  # 1-indexed
        profile_id = row['profile_id']  # The profile ID is used as the key in the JSON
        
        # Construct the filename based on the row data
        json_filename = f"{model}_{temperature:.1f}_{scenario}.json"
        json_filepath = os.path.join(base_path, json_filename)
        
        # Check if the file exists before attempting to read
        if os.path.exists(json_filepath):
            with open(json_filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if the profile ID exists in the data
            profile_key = str(profile_id)  # Convert profile_id to string for matching
            if profile_key in data:
                profile_data = data[profile_key]
                
                # Ensure the sample is within the valid range of rows for this profile
                if 1 <= sample <= len(profile_data):
                    # Delete the problematic sample (1-indexed, so we subtract 1)
                    del profile_data[sample - 1]
                    
                    # If overwrite is True, we save the modified data back to the JSON file
                    if overwrite:
                        with open(json_filepath, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                    else:
                        # Save as a cleaned version with "_clean" suffix
                        with open(json_filepath.replace('.json', '_clean.json'), 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=4)
                    
                    print(f"Deleted sample {sample} (profile ID {profile_id}) from {json_filename}")
                else:
                    print(f"Sample {sample} is out of range for profile ID {profile_id} in {json_filename}")
            else:
                print(f"Profile ID {profile_id} not found in {json_filename}")
        else:
            print(f"File {json_filename} not found.")
