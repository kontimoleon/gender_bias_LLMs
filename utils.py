import os
import json
import yaml
import logging
from datetime import datetime


def load_yaml(file_path):
    with open(file_path) as f:
        config = yaml.safe_load(f)
    return config


def load_json(file_path):
    with open(file_path) as f:
        json_data = json.load(f)
    return json_data


def save_json(data, file_path):
    """Write data to a JSON file with an option to create the file or update the dictionary."""
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        with open(file_path, 'r') as file:
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
        
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)


def combine_json_files(input_files: list, output_file: str):
    for file in input_files:
        json_data = load_json(file)
        save_json(json_data, output_file)


def combine_json_files_efficient(input_files: list, output_file: str):
    combined_data = {}
    for file in input_files:
        json_data = load_json(file)
        for key, value in json_data.items():
            if key in combined_data:
                combined_data[key].extend(value)
            else:
                combined_data[key] = list(value)  # creates a shallow copy of value, given that it's a list
    save_json(combined_data, output_file)


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