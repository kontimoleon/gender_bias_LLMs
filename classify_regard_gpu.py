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

def classify_regard(dataset):
    batch_results = []
    for entry in tqdm(dataset):
        try:
            texts = entry["text"]
            avg_result = regard.compute(data=texts, aggregation="average")['average_regard']
            profile_regard = {
                "profile_id": entry["profile_id"],
                "gender": "male" if entry["profile_id"] in male_profiles else "female",
                "positive_regard": avg_result['positive'],
                "negative_regard": avg_result['negative'],
                "neutral_regard": avg_result['neutral'],
                "other_regard": avg_result['other']
            }
            batch_results.append(profile_regard)
        except Exception as e:
            logging.error(f"Error computing regard for profile {entry['profile_id']}: {e}")
    return batch_results

if __name__ == "__main__":
    configure_logging(script_name="classify_regard")
    try:
        # Load config
        logging.info("Loading configuration...")
        config = load_yaml('config/analysis_config.yaml')['regard']
        
        # Load and filter data
        filtered_df = load_and_filter_data(config)
        logging.info(f"Total profiles to process: {len(filtered_df)}")

        # Convert pandas DataFrame to Hugging Face Dataset
        dataset = Dataset.from_pandas(filtered_df[['profile_id', 'text']])
        
        # Batch processing
        batch_size = config['batch_size']
        result_list = []
        
        start_batch = config['start_batch']
        start_index = start_batch * batch_size

        # Process the data in batches
        for i in range(start_index, len(dataset), batch_size):
            batch_data = dataset[i:i+batch_size]
            current_batch_size = i // batch_size + 1
            logging.info(f"Processing batch {current_batch_size}...")

            # Apply sentiment analysis to the batch
            batch = classify_regard(batch_data)

            result_list.append(batch)
            
            # Save intermediate batch
            save_batch(batch, config['model_name'], config['scenario'], current_batch_size)
        
        # Save final results
        save_final_result(result_list, config['model_name'], config['scenario'])
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
