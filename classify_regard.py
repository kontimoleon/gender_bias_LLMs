import os
import pickle
import evaluate 
import logging
import pandas as pd

from tqdm import tqdm
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

def classify_regard(df, batch_ids):
    batch_results = []
    for pid in tqdm(batch_ids):
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
    return pd.concat(batch_results)

if __name__ == "__main__":
    configure_logging(script_name="classify_regard")
    try:
        # Load config
        logging.info("Loading configuration...")
        config = load_yaml('config/analysis_config.yaml')['regard']
        
        # Load and filter data
        filtered_df = load_and_filter_narrative_dataframe(config)
        profile_ids = sorted(filtered_df['profile_id'].unique())
        logging.info(f"Total profiles to process: {len(profile_ids)}")
        
        # Batch processing
        batch_size = config['batch_size']
        result_list = []
        
        start_batch = config['start_batch']
        start_index = start_batch * batch_size

        # Process the data in batches
        for i in range(0, len(profile_ids), batch_size):
            batch_ids = profile_ids[i:i+batch_size]
            tqdm.pandas()
            current_batch_size = i // batch_size + 1
            logging.info(f"Processing batch {current_batch_size}...")

            # Apply sentiment analysis to the batch
            batch = classify_regard(filtered_df, batch_ids)

            result_list.append(batch)
            
            # Save intermediate batch
            save_batch(batch, config['model_name'], config['scenario'], current_batch_size)
        
        # Save final results
        save_final_result(result_list, config['model_name'], config['scenario'])
        
    except Exception as e:
        logging.error(f"An error occurred: {e}")
