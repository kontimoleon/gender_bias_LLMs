import os
import logging

from tqdm import tqdm
from openai import OpenAI

from utils import load_json, save_json, load_yaml, configure_logging

def set_up_response(model_name, prompt):
    model_id = load_yaml("./config/model_config.yaml")[model_name]['model_id']
    if model_name == "gpt":
        response = opeanAI_request(prompt, model_id)
    return response


def opeanAI_request(prompt, model_id):
    # TODO: add temperature to the request
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_id,
        store=True,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response


def generate_single_narrative(model_name, prompt):
    response = set_up_response(model_name, prompt)
    return response.choices[0].message.content


def generate_narratives_for_model(overwrite, model_name, gender_scenarios):
    logging.info(f"Starting narrative generation for model '{model_name}'.")

    for gender_scenario in gender_scenarios:
        model_id = load_yaml("./config/model_config.yaml")[model_name]['model_id']
        output_file = f"./data/narratives/{model_id}_gender_{gender_scenario}.json"

        logging.info(f"Processing gender scenario '{gender_scenario}'.")
        
        if os.path.exists(output_file) and not overwrite:
            logging.info(f"Output file '{output_file}' exists. Filling in missing narratives.")
            all_profiles = list(load_json('./data/synthetic_profiles.json').keys())
            existing_profiles = list(load_json(output_file).keys())
            profiles = list(set(all_profiles) - set(existing_profiles))
        else:
            logging.info(f"Starting fresh or overwriting existing file '{output_file}'.")
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            profiles = list(load_json('./data/synthetic_profiles.json').keys())

        prompts = load_json(f'./data/prompts_gender_{gender_scenario}.json')

        # for profile in tqdm(profiles, desc=f"Generating narratives"):
        for profile in tqdm(['9', '256', '1012'], desc=f"Generating narratives"):
            try:
                prompt = next(
                    filter(lambda item: item['profile_id'] == profile, prompts)
                )['prompt_text']

                narrative_text = generate_single_narrative(model_name, prompt)
                narrative_entry = {profile: narrative_text}
                save_json(narrative_entry, output_file, overwrite)
                logging.info(f"Narrative generated and saved for profile '{profile}'.")
            except StopIteration:
                logging.error(f"No prompt found for profile '{profile}'. Skipping.")
            except Exception as e:
                logging.error(f"Error generating narrative for profile '{profile}': {e}")

def generate_narratives(overwrite, model_names, gender_scenarios):
    logging.info("Starting narrative generation for all models.")

    for model_name in tqdm(model_names, desc="Processing models"):
        generate_narratives_for_model(overwrite, model_name, gender_scenarios)


if __name__ == "__main__":
    configure_logging(script_name='generate_narratives')

    config = load_yaml("./config/narrative_config.yaml")

    overwrite = config['overwrite']
    model_names = [model for model in list(config['models'].keys()) if config['models'][model] == True]
    gender_scenarios = [gender for gender, enabled in config["gender_scenarios"].items() if enabled]

    generate_narratives(overwrite, model_names, gender_scenarios)

    logging.info("Narrative generation completed.")
