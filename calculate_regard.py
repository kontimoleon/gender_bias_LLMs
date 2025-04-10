import os
import pickle
import evaluate 
import logging
import pandas as pd

from tqdm import tqdm
from utils import configure_logging

# ========== SETTINGS ========== #
BATCH_SIZE = 50
SAVE_DIR = "data/analysis/regard_batches"
os.makedirs(SAVE_DIR, exist_ok=True)

# ========== DATA SOURCES ========== #
narrative_dict = {
    "given": {
        "gpt": [
            "gpt-4o-mini_0.0_given.json",
            "gpt-4o-mini_0.5_given.json",
            "gpt-4o-mini_1.2_given.json"
        ],
        "llama": [
            "llama3.3-b_0.0_given.json",
            "llama3.3-b_0.5_given.json",
            "llama3.3-b_1.0_given.json"
        ]
    }
}

regard = evaluate.load("regard")

male_profiles = [i + 1 for i in range(528)]
female_profiles = [i + 528 for i in male_profiles]

# ========== LOAD + FORMAT ========== #
def load_json_narratives_to_df(narr_dir, narr_files):
    df_list = []
    for file in tqdm(narr_files, desc="Loading narrative files"):
        logging.info(f"Loading file: {file}")
        try:
            df = pd.read_json(os.path.join(narr_dir, file))
            formatted_df = format_narrative_df(df, file)
            df_list.append(formatted_df)
        except Exception as e:
            logging.error(f"Failed to load or format {file}: {e}")
    return pd.concat(df_list, ignore_index=True)

def format_narrative_df(df, file_name):
    df_formatted = df.melt(var_name='profile_id', value_name='text')
    df_formatted['model'] = file_name.split('_')[0]
    df_formatted['temperature'] = float(file_name.split('_')[1])
    df_formatted['scenario'] = file_name.split('_')[2].split('.')[0]
    df_formatted['sample'] = df_formatted.groupby('profile_id').cumcount() + 1
    return df_formatted

# ========== CALCULATE REGARD ========== #
def calculate_regard(df, profile_ids, model, scenario):
    logging.info(f"Total profiles to process: {len(profile_ids)}")
    regard_results = []

    for i in range(0, len(profile_ids), BATCH_SIZE):
        batch_ids = profile_ids[i:i+BATCH_SIZE]
        batch_results = []

        for pid in tqdm(batch_ids, desc=f"Batch {i//BATCH_SIZE+1}"):
            try:
                texts = df[df["profile_id"] == pid]["text"].tolist()
                avg_result = regard.compute(data=texts, aggregation="average")['average_regard']
                profile_regard = pd.DataFrame([{
                    "profile_id": pid,
                    "gender": "male" if pid in male_profiles else "female",
                    "positive_regard": avg_result['positive'],
                    "negative_regard": avg_result['negative'],
                    "neutral_regard": avg_result['neutral'],
                    "other_regard": avg_result['other']
                }])
                batch_results.append(profile_regard)
            except Exception as e:
                logging.error(f"Error computing regard for profile {pid}: {e}")

        batch_df = pd.concat(batch_results)
        temp_path = os.path.join(SAVE_DIR, f"{model}_{scenario}_batch_{i//BATCH_SIZE+1}.pkl")
        with open(temp_path, "wb") as f:
            pickle.dump(batch_df, f)
        logging.info(f"Saved batch {i//BATCH_SIZE+1} to {temp_path}")
        regard_results.append(batch_df)

    # Combine all batches
    final_df = pd.concat(regard_results)
    output_path = f"data/analysis/{model}_{scenario}_avg_regard_per_profile.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(final_df, f)
    logging.info(f"Saved final regard results to: {output_path}")

# ========== MAIN ========== #
if __name__ == "__main__":
    configure_logging(script_name="calculate_regard")

    for scenario in narrative_dict:
        for model in narrative_dict[scenario]:
            logging.info(f"\nProcessing model: {model} | scenario: {scenario}")
            narrs = load_json_narratives_to_df(
                narr_dir='data/narratives/',
                narr_files=narrative_dict[scenario][model]
            )
            profile_ids = sorted(narrs["profile_id"].unique())
            calculate_regard(narrs, profile_ids, model, scenario)
