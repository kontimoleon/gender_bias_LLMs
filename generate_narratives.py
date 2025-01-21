import os

import aisuite as ai
client = ai.Client()

from tqdm import tqdm
from utils import load_json, save_json, load_config, configure_logging


def set_up_response(model_name, prompt):
    model_config = load_config("./config/model_config.yaml")[model_name]
    response = client.chat.completions.create(
        model= f"{model_config['provider']}:{['model_id']}" if model_name in ['gpt', 'claude'] else model_config['model_id'],
        messages=[
            {"role": "user", "content": prompt},
        ]
    )
    return response


def generate_single_narrative(model_name, prompt):
    response = set_up_response(model_name, prompt)
    return response.choices[0].message.content


def generate_narratives_for_template(overwrite, model_name, template, gender_scenario):
    output_file = f"./data/narratives/{model_name}_gender_{gender_scenario}_{template}.json"
    # first we check if the file exists
    if os.path.exists(output_file) and not overwrite:
        # if it does and we're not meant to overwrite it,
        # it means that we're filling in missing narratives
        all_profiles = list(load_json('./data/synthetic_profiles.json').keys())
        existing_profiles = load_json(output_file).keys()
        profiles = list(set(all_profiles) - set(existing_profiles))
    else:
        profiles = list(load_json('./data/synthetic_profiles.json').keys())
    
    prompts = load_json(f'./data/prompts_gender_{gender_scenario}.json')[template]
    for profile in profiles:
        prompt = next( # for lazy evaluation
            # filter the prompts to the one that corresponds to the profile
            filter(lambda item: item['profile_id'] == profile, prompts)
        )['prompt_text']
        narrative_text = generate_single_narrative(model_name, prompt)
        narrative_entry = {
            profile: narrative_text
        }
        save_json(narrative_entry, output_file, overwrite)


def generate_narratives_for_model(overwrite, model_name, templates, gender_scenarios):
    for gender_scenario in gender_scenarios:
        for template in templates:
            generate_narratives_for_template(overwrite, model_name, template, gender_scenario)


def generate_narratives(overwrite, model_names, templates, gender_scenarios):
    for model_name in model_names:
        generate_narratives_for_model(overwrite, model_name, templates, gender_scenarios)


if __name__ == "__main__":
    configure_logging(script_name='generate_narratives')
    config = load_config("./config/narrative_config.yaml")

    overwrite = config['overwrite']
    model_names = [model for model in list(config['models'].keys()) if config['models'][model] == True]
    templates = [template for template in list(config['templates'].keys()) if config['templates'][template] == True]
    gender_scenarios = [gender for gender, enabled in config["gender_scenarios"].items() if enabled]

    # Call the function with the parsed arguments
    generate_narratives(overwrite, model_names, templates, gender_scenarios)
