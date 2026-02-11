import os
import pickle
import torch
import logging
import pandas as pd

from tqdm import tqdm
from datasets import Dataset
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from scripts.utils import load_yaml, configure_logging


def predict_robust_sentiment(text, model, tokenizer, device):
    """
    Predict sentiment using the robust sentiment model on GPU or CPU.
    """
    inputs = tokenizer(text.lower(), return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to device
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    return sentiment_map[predicted_class]


def initialize_sentiment_model(config):
    """
    Initialize sentiment model based on the configuration.
    """
    device = 0 if torch.cuda.is_available() else -1
    model_type = config['sentiment_model']
    
    if model_type == 'siebert':
        logging.info("Initializing SiEBERT sentiment model with GPU support...")
        sentiment_model = pipeline("sentiment-analysis", model="siebert/sentiment-roberta-large-english", device=device)
        return sentiment_model, None, device
    
    elif model_type == 'robust':
        logging.info("Initializing Robust sentiment model with GPU support...")
        model_name = "tabularisai/robust-sentiment-analysis"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        return sentiment_model, tokenizer, device
    
    elif model_type == 'bert-base-multilingual':
        logging.info("Initializing BERT-base Multilingual Uncased sentiment model with GPU support...")
        model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        sentiment_model = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)
        return sentiment_model, None, device

    else:
        logging.error("Sentiment model not recognized.")
        raise ValueError("Sentiment model not recognized.")


def load_and_filter_narrative_dataframe(config):
    """
    Load the clean narratives DataFrame and filter based on the config (model and scenario only).
    """
    logging.info("Loading narrative data...")
    with open('data/narratives/narrative_df_clean.pkl', 'rb') as f:
        df = pickle.load(f)
    
    logging.info("Filtering data based on model and scenario...")
    model_name = config['model_name']
    scenario = config['scenario']
    logging.info(f"Model: {model_name} | Scenario: {scenario}")
    
    filtered_df = df[(df['model'] == model_name) & (df['scenario'] == scenario)]
    
    return filtered_df


def process_batch(batch, sentiment_model, tokenizer=None, device=None, model_type='siebert'):
    """
    Process a batch of data and apply sentiment analysis.
    """
    tqdm.pandas()
    if sentiment_model is None:
        logging.error("Sentiment model is None. Skipping batch.")
        return batch

    if model_type == 'robust':
            logging.info("Processing Robust sentiment one-by-one using GPU inference...")
            batch['robust_sentiment'] = batch['text'].progress_apply(
                lambda x: predict_robust_sentiment(x, sentiment_model, tokenizer, device)
            )

    else:
        logging.info("Processing sentiment using batched HF Dataset pipeline...")
        # Convert to HF dataset with a "text" column
        dataset = Dataset.from_pandas(batch[['text']].reset_index(drop=True))

        # Apply sentiment pipeline in batches
        predictions = sentiment_model(dataset['text'], batch_size=32)

        # Extract labels and assign back
        batch['siebert'] = [p['label'] for p in predictions]

    return batch


def save_batch(batch, model_name, scenario, sentiment_model_name, current_batch_size):
    """
    Save the processed batch to a pickle file inside a temp folder.
    """
    temp_dir = "data/analysis/temp"
    os.makedirs(temp_dir, exist_ok=True)
    output_filename = os.path.join(temp_dir, f"{model_name}_{scenario}_{sentiment_model_name}_temp_batch_{current_batch_size}.pkl")
    
    with open(output_filename, 'wb') as f:
        pickle.dump(batch, f)


def save_final_result(result_list, model_name, scenario, sentiment_model_name):
    """
    Save the final processed DataFrame.
    """
    final_df = pd.concat(result_list, ignore_index=True)
    final_filename = f"data/analysis/{model_name}_{scenario}_{sentiment_model_name}.pkl"
    logging.info(f"Saving final results to {final_filename}...")
    with open(final_filename, 'wb') as f:
        pickle.dump(final_df, f)


if __name__ == "__main__":
    configure_logging(script_name='classify_sentiment_gpu')
    try:
        # Load config
        logging.info("Loading configuration...")
        config = load_yaml('config/analysis_config.yaml')['sentiment']
        
        # Load and filter data
        filtered_df = load_and_filter_narrative_dataframe(config)
        
        # Initialize sentiment model
        sentiment_model, tokenizer, device = initialize_sentiment_model(config)
        model_type = config['sentiment_model']

        # Batch processing
        batch_size = config['batch_size']
        result_list = []
        start_batch = config['start_batch']
        start_index = start_batch * batch_size

        for i in range(start_index, len(filtered_df), batch_size):
            batch = filtered_df.iloc[i:i+batch_size].copy()
            current_batch_size = i // batch_size + 1
            logging.info(f"Processing batch {current_batch_size}...")

            # Apply sentiment analysis
            batch = process_batch(batch, sentiment_model, tokenizer, device, model_type)

            result_list.append(batch)

            # Save intermediate result
            save_batch(batch, config['model_name'], config['scenario'], model_type, current_batch_size)

        # Save final DataFrame
        save_final_result(result_list, config['model_name'], config['scenario'], model_type)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
