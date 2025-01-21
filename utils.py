import os
import json
import yaml
import logging
import datetime


def load_config(file_path):
    with open(file_path) as f:
        config = yaml.safe_load(f)
    return config


def load_json(file_path):
    with open(file_path) as f:
        json_data = json.load(f)
    return json_data


def save_json(data, file_path, overwrite = False):
    """Write data to a JSON file with the option to overwrite or append."""
    if overwrite or not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    else:
        with open(file_path, 'r') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []  # Start fresh if the file is empty or corrupt
        existing_data.extend(data)
        with open(file_path, 'w') as file:
            json.dump(existing_data, file, indent=4)


def configure_logging(script_name):
    log_file_path = os.path.dirname('./logs')
    os.makedirs(log_file_path, exist_ok=True)
    
    log_filename = f"{log_file_path}/{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filename=log_filename,
        filemode="w"
    )