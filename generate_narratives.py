import os
import time
import logging

from tqdm import tqdm
from openai import OpenAI

from utils import load_json, save_json, load_yaml, configure_logging

def set_up_response(model_name, prompt, temp_value):
    model_id = load_yaml("./config/model_config.yaml")[model_name]['model_id']

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
        temperature_values,
        samples_per_profile,
        starting_id
    ):
    logging.info(f"Starting narrative generation for model '{model_name}'.")

    for temp_value in temperature_values:
        logging.info(f"Initiating generation with temperature set to '{temp_value}'.")

        for gender_scenario in gender_scenarios:
            model_id = load_yaml("./config/model_config.yaml")[model_name]['model_id']

            logging.info(f"Processing gender scenario '{gender_scenario}'.")
            output_file = f"./data/narratives/{model_id}_temp{temp_value}_gender_{gender_scenario}.json"
            
            if os.path.exists(output_file):
                logging.info(f"Output file '{output_file}' exists. Appending narratives.")
            else:
                logging.info(f"Output file: '{output_file}' does not exist. Starting fresh.")
                os.makedirs(os.path.dirname(output_file), exist_ok=True) 

            prompts = load_json(f'./data/prompts_gender_{gender_scenario}.json')
            if starting_id != 1:
                # in case we're filling in narratives we only retrieve the relevant profiles
                profiles = list(load_json('./data/synthetic_profiles.json').keys())[(starting_id-1):]
                logging.warning(f"You're starting the generation from profile {starting_id}.")
            else:
                profiles = list(load_json('./data/synthetic_profiles.json').keys())
            
            logging.info(f"Generating {samples_per_profile} narrative(s) per profile for {len(profiles)} profiles.")
            logging.info(f"Model ID: {model_id}. Gender scenario: {gender_scenario}. Temperature: {temp_value}.")
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
                logging.info(f"Narrative generation round {i+1} completed in {(end - start)/3600} hours.")

def generate_narratives(
        model_name,
        gender_scenarios,
        temperature_values,
        samples_per_profile,
        starting_id = 1
    ):
    logging.info("Starting narrative generation for all models.")

    for model_name in tqdm(model_names, desc="Processing models"):
        generate_narratives_for_model(
            model_name,
            gender_scenarios,
            temperature_values,
            samples_per_profile,
            starting_id
        )


if __name__ == "__main__":
    configure_logging(script_name='generate_narratives')

    config = load_yaml("./config/narrative_config.yaml")

    model_names = [model for model in list(config['models'].keys()) if config['models'][model] == True]
    gender_scenarios = [gender for gender, enabled in config["gender_scenarios"].items() if enabled]
    temperature_values = config['temperature_values']
    samples_per_profile = config['samples_per_profile']
    starting_id = config['starting_profile_id']

    kwargs = {}
    if starting_id is not None:
        kwargs["starting_id"] = starting_id

    generate_narratives(model_names, gender_scenarios, temperature_values, samples_per_profile, **kwargs)

    logging.info("Narrative generation completed.")
