import os
import time
import logging

from tqdm import tqdm
from openai import OpenAI

from utils import load_json, save_json, load_yaml, configure_logging

def set_up_response(model_name, prompt, temp_value):
    model_id = load_yaml("./config/narrative_config.yaml")['models'][model_name]['model_id']

    # set-up OpenAI client accordingly
    if model_name == "gpt":
        client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name in ["deepseek", "llama"]:
        client = OpenAI(
            base_url="https://gpu.gess-k8s.ethz.ch/v1-openai", 
            api_key=os.getenv("GPUSTACK_API_KEY")
        )

    # make request to client to retrieve chat completion response
    response = client.chat.completions.create(
        model=model_id,
        store=False,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp_value
    )
    
    return response.choices[0].message.content

def generate_narrative_for_profile(profile, model_name, prompt, temp_value):
    narrative_text = set_up_response(model_name, prompt, temp_value)
    narrative_entry = {profile: [narrative_text]}
    return narrative_entry

def generate_narratives_for_model(
        model_name,
        gender_scenarios,
        samples_per_profile,
        **kwargs
    ):
    logging.info(f"Starting narrative generation for model '{model_name}'.")

    # load all profiles once per model
    all_profiles = list(load_json('./data/synthetic_profiles.json').keys())

    # Retrieve optional arguments with defaults
    first_id = kwargs.get("first_id", 1)
    last_id = kwargs.get("last_id") #implicitly defaults to None
    output_suffix = kwargs.get("output_suffix")

    # check validity of user-input arguments
    if not (1 <= first_id <= len(all_profiles)):
        raise ValueError(f"first_profile_id {first_id} out of range. Please adjust the configuration file.")
    if last_id is not None and not (1 <= last_id <= len(all_profiles)):
        raise ValueError(f"last_profile_id {last_id} out of range. Please adjust the configuration file.")
    if last_id is not None and last_id < first_id:
        raise ValueError(f"first_id ({first_id}) must be <= last_id ({last_id}). Please adjust the configuration file.")


    # Load config for specific model
    model_config = load_yaml("./config/narrative_config.yaml")['models']
    model_id = model_config[model_name]['model_id']
    model_temp_values = model_config[model_name]['temperature_values']

    for temp_value in model_temp_values:
        for gender_scenario in gender_scenarios:
            logging.info(f"Model ID: {model_id}. Gender scenario: {gender_scenario}. Temperature: {temp_value}.")

            output_file = f"./data/narratives/{model_id}_{temp_value}_{gender_scenario}"
            if output_suffix:
                output_file += f"_{output_suffix}"
            output_file += ".json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            logging.info(f"Writing to {output_file}.")

            prompts = load_json(f'./data/prompts_gender_{gender_scenario}.json')

            # Apply profile slicing, in case we're filling in narratives
            first_idx = first_id-1 # profiles are 1-indexed
            last_idx = last_id if last_id is not None else len(all_profiles) # slicing is exclusive on the upper bound
            # retrieving relevant profiles
            profiles = all_profiles[first_idx:last_idx] 
            
            logging.warning(f"Processing profiles from {first_id} to {last_id if last_id is not None else 'end'}.")
            logging.info(f"Generating {samples_per_profile} narrative(s) per profile for {len(profiles)} profiles.")

            for i in tqdm(range(samples_per_profile), desc=f"Generating narratives."):
                logging.info(f"Starting narrative generation, round {i+1}.")
                start = time.time()
                for profile in tqdm(profiles):
                    try:
                        prompt = next(
                            filter(lambda item: item['profile_id'] == profile, prompts)
                        )['prompt_text']

                        narrative_text = generate_narrative_for_profile(profile, model_name, prompt, temp_value)
                        save_json(narrative_text, output_file)
                        # comment out to make log less verbose?
                        # logging.info(f"Narrative generated and saved for profile '{profile}'.")
                    except StopIteration:
                        logging.error(f"No prompt found for profile '{profile}'. Skipping.")
                    except Exception as e:
                        logging.error(f"Error generating narrative for profile '{profile}': {e}")
                end = time.time()
                logging.info(f"Narrative generation round {i+1} completed in {(end - start)/3600:.2f} hours.")

def generate_narratives(
        model_names,
        gender_scenarios,
        samples_per_profile,
        **kwargs
    ):
    logging.info("Starting narrative generation for all models.")

    for model_name in model_names:
        generate_narratives_for_model(
            model_name,
            gender_scenarios,
            samples_per_profile,
            **kwargs
        )


if __name__ == "__main__":
    configure_logging(script_name='generate_narratives')

    config = load_yaml("./config/narrative_config.yaml")

    model_names = [model for model in list(config['models'].keys()) if config['models'][model]['include'] == True]
    gender_scenarios = [gender for gender, enabled in config["gender_scenarios"].items() if enabled]
    samples_per_profile = config['samples_per_profile']
    first_id = config['first_profile_id']
    last_id = config['last_profile_id']
    output_suffix = config['output_suffix']

    kwargs = {}
    if first_id is not None:
        kwargs["first_id"] = first_id
    if last_id is not None:
        kwargs["last_id"] = last_id
    if output_suffix is not None:
        kwargs["output_suffix"] = output_suffix

    generate_narratives(model_names, gender_scenarios, samples_per_profile, **kwargs)

    logging.info("Narrative generation completed.")
