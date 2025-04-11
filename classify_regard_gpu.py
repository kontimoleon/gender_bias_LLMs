import os
import pickle
import evaluate
import logging
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from utils import configure_logging, load_yaml
from classify_sentiment import load_and_filter_data

regard = evaluate.load("regard")

male_profiles = [i + 1 for i in range(528)]
female_profiles = [i + 528 for i in male_profiles]

def save_batch(batch, model_name, scenario, current_batch_size):
    """
    Save the processed batch to a pickle file inside a temp folder.
    """
    temp_dir = "data/analysis/temp"
    os.makedirs(temp_dir, exist_ok=True)  # Create temp directory if it doesn't exist

    output_filename = os.path.join(temp_dir, f"{model_name}_{scenario}_regard_temp_batch_{current_batch_size}.pkl")
    
    with open(output_filename, 'wb') as f:
        pickle.dump(batch, f)

def save_final_result(result_list, model_name, scenario):
    """
    Save the final processed DataFrame.
    """
    final_df = pd.concat(result_list, ignore_index=True)
    final_filename = f"data/analysis/{model_name}_{scenario}_avg_regard_per_profile.pkl"
    logging.info(f"Saving final results to {final_filename}...")
    with open(final_filename, 'wb') as f:
        pickle.dump(final_df, f)

def classify_regard(profile_data):
    """
    Classify regard for a single profile.
    """
    # Use Hugging Face pipeline to process the single profile's texts
    texts = profile_data["text"]
    avg_result = regard.compute(data=texts, aggregation="average")['average_regard']
    
    # Prepare the result
    profile_regard = {
        "profile_id": profile_data["profile_id"][0],
        "gender": "male" if profile_data["profile_id"][0] in male_profiles else "female",
        "positive_regard": avg_result['positive'],
        "negative_regard": avg_result['negative'],
        "neutral_regard": avg_result['neutral'],
        "other_regard": avg_result['other']
    }

    # Return the result as a DataFrame
    return pd.DataFrame([profile_regard])

if __name__ == "__main__":
    configure_logging(script_name="classify_regard")
    try:
        # Load config
        logging.info("Loading configuration...")
        config = load_yaml('config/analysis_config.yaml')['regard']
        
        # Load and filter data
        filtered_df = load_and_filter_data(config)
        profile_ids = sorted(filtered_df['profile_id'].unique())
        logging.info(f"Total profiles to process: {len(profile_ids)}")
        
        # Convert pandas DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(filtered_df[['profile_id', 'text']])
        
        result_list = []
        
        # Process each profile separately
        for profile_id in profile_ids:
            # Filter the dataset to include only the current profile's data
            profile_data = dataset.filter(lambda x: x["profile_id"] == profile_id)

            logging.info(f"Processing profile {profile_id}...")

            # Apply the classification function to this profile
            batch = classify_regard(profile_data)

            result_list.append(batch)
            
            # Save intermediate batch (optional, can be skipped)
            save_batch(batch, config['model_name'], config['scenario'], profile_id)
        
        # Save final results
        save_final_result(result_list, config['model_name'], config['scenario'])
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
