import os
import pickle
import evaluate 
import pandas as pd

from tqdm import tqdm

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

def load_json_narratives_to_df(narr_dir, narr_files):
    df_list = [
        format_narrative_df(pd.read_json(os.path.join(narr_dir, file)), file)
        for file in tqdm(narr_files, desc="Loading narrative files")
    ]
    narr_df = pd.concat(df_list, ignore_index=True)
    return narr_df

def format_narrative_df(df, file_name):
    print(f"Formatting: {file_name}")
    df_formatted = df.melt(var_name='profile_id', value_name='text')
    df_formatted['model'] = file_name.split('_')[0]
    df_formatted['temperature'] = float(file_name.split('_')[1])
    df_formatted['scenario'] = file_name.split('_')[2].split('.')[0]
    df_formatted['sample'] = df_formatted.groupby('profile_id').cumcount() + 1
    return df_formatted

def calculate_regard(df, profile_ids, model, scenario):
    regard_results = []
    for pid in tqdm(profile_ids, desc=f"Calculating regard for {model} ({scenario})"):
        texts = df[df["profile_id"] == pid]["text"].tolist()
        avg_result = regard.compute(data=texts, aggregation="average")['average_regard']
        profile_regard = pd.DataFrame(
            data=[{
                "profile_id": pid,
                "gender": "male" if pid in male_profiles else "female",
                "positive_regard": avg_result['positive'],
                "negative_regard": avg_result['negative'],
                "neutral_regard": avg_result['neutral'],
                "other_regard": avg_result['other']
            }]
        )
        regard_results.append(profile_regard)

    # Save results as a DataFrame
    regard_df = pd.concat(regard_results)
    output_path = f"data/analysis/{model}_{scenario}_avg_regard_per_profile.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(regard_df, f)
    print(f"Saved regard results to: {output_path}")

if __name__ == "__main__":
    for scenario in narrative_dict:
        for model in narrative_dict[scenario]:
            print(f"\nProcessing model: {model} | scenario: {scenario}")
            narrs = load_json_narratives_to_df(
                narr_dir='data/narratives/',
                narr_files=narrative_dict[scenario][model]
            )
            profile_ids = sorted(narrs["profile_id"].unique())
            calculate_regard(narrs, profile_ids, model, scenario)
