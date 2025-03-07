import os
from datetime import datetime
from utils import load_json, append_to_txt_file


def inspect_narratives(narr_dir: str, output_file: str):
    for filename in os.listdir(narr_dir):
        if filename.endswith(".json"):
            narr_path = os.path.join(narr_dir, filename)
            narr_data = load_json(narr_path)  # Load JSON file
            narr_stats = narrative_statistics(narr_data)  # Analyze statistics
            write_report_for_narr_file(narr_stats, filename, output_file)  # Write results


def write_report_for_narr_file(narr_stats, filename: str, output_file: str):
    no_profiles, consistent_profiles, inconsistent_profiles = narr_stats

    entry = (
        f"Inspecting the data for {filename}\n"
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        "\n"
        f"Found {no_profiles} profiles in that file.\n"
        f"Found {len(consistent_profiles)} profiles with consistent narratives.\n"
    )

    if inconsistent_profiles:
        entry += "The following profiles had irregular numbers of narratives:\n"
        for pid, counts in inconsistent_profiles.items():
            entry += f"Profile {pid} has {counts} different narrative counts across files.\n"

    entry += "-----------------------------\n"

    append_to_txt_file(entry, output_file)  # Write to output file


def narrative_statistics(narr_data):
    # We start by counting the number of profiles in the file
    no_profiles = len(narr_data.keys())

    # Count the number of narratives per profile
    narr_counts = {}
    for profile_id, narratives in narr_data.items():
        count = len(narratives)
        if profile_id not in narr_counts:
            narr_counts[profile_id] = {count}  # Store in a set to track unique counts
        else:
            narr_counts[profile_id].add(count)

    # Identify consistent and inconsistent profiles
    consistent_profiles = {pid: next(iter(counts)) for pid, counts in narr_counts.items() if len(counts) == 1}
    inconsistent_profiles = {pid: counts for pid, counts in narr_counts.items() if len(counts) > 1}

    return no_profiles, consistent_profiles, inconsistent_profiles


if __name__ == "__main__":
    narr_dir = "./data/narratives"
    output_dir = "./data/inspections"
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamped report file
    output_file = os.path.join(output_dir, f"narrative_inspection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    # Initialize the output file
    with open(output_file, "w") as f:
        f.write("")

    inspect_narratives(narr_dir, output_file)
