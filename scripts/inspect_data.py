import os
from datetime import datetime
from scripts.utils import load_json, load_yaml, append_to_txt_file


def inspect_narratives(narr_dir: str, output_file: str, ends_with:str):
    for filename in os.listdir(narr_dir):
        if filename.endswith(ends_with):
            narr_path = os.path.join(narr_dir, filename)
            narr_data = load_json(narr_path)  # Load JSON file
            narr_stats = narrative_statistics(narr_data)  # Analyze statistics
            write_report_for_narr_file(narr_stats, filename, output_file)  # Write results


def write_report_for_narr_file(narr_stats, filename: str, output_file: str):
    no_profiles, unique_counts, inconsistencies = narr_stats

    entry = (
        "\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Inspecting the data for {filename}\n"
        "\n"
        f"Found {no_profiles} profiles in that file.\n"
      )

    if len(unique_counts)==1:
        entry += f"All profiles have the same number of narratives: {unique_counts}\n"
    else:
        entry += f"Found inconsistent numbers of narratives across profiles.\n"
        for k, v in inconsistencies.items():
            entry += (
                f"{len(v)} profiles have {k} varratives each.\n"
                f"Their IDs are: {[i for i in v]}\n"
            )

    entry += "-----------------------------\n"

    append_to_txt_file(entry, output_file)  # Write to output file


def narrative_statistics(narr_data):
    # We start by counting the number of profiles in the file
    no_profiles = len(narr_data.keys())

    # Count the number of narratives per profile
    narr_counts = {}
    unique_counts = set()
    for profile_id, narratives in narr_data.items():
        count = len(narratives)
        narr_counts[profile_id] = count  # Store in a set to track unique counts
        unique_counts.add(count)

    inconsistencies = {}
    if len(unique_counts) != 1:
        for profile_id, count in narr_counts.items():
          if count in inconsistencies.keys():
              inconsistencies[count].append(profile_id)
          else:
              inconsistencies[count] = [profile_id]

    return no_profiles, unique_counts, inconsistencies


if __name__ == "__main__":
    inspection_config = load_yaml("./config/inspection_config.yaml")
    narr_dir = inspection_config["narr_dir"]
    output_dir = inspection_config["output_dir"]
    ends_with = inspection_config["ends_with"]
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamped report file
    output_file = os.path.join(output_dir, f"narrative_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Initialize the output file
    with open(output_file, "w") as f:
        f.write("")

    inspect_narratives(narr_dir, output_file, ends_with)
