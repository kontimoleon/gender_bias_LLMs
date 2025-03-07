import os
import time
import logging
import asyncio

from tqdm import tqdm
from openai import AsyncOpenAI

from utils import load_json, save_json, load_yaml, configure_logging

async def set_up_response(model_name, prompt, temp_value):
    model_id = load_yaml("./config/model_config.yaml")[model_name]['model_id']

    # set-up AsyncOpenAI client accordingly
    if model_name == "gpt":
        client = AsyncOpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
    elif model_name in ["deepseek", "llama"]:
        client = AsyncOpenAI(
            base_url="https://gpu.gess-k8s.ethz.ch/v1-openai", 
            api_key=os.getenv("GPUSTACK_API_KEY")
        )

    # make request to client to retrieve chat completion response
    response = await client.chat.completions.create(
        model=model_id,
        store=False,
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=temp_value
    )
    
    return response.choices[0].message.content

async def generate_narrative_for_profile(profile, model_name, prompt, temp_value):
    narrative_text = await set_up_response(model_name, prompt, temp_value)
    narrative_entry = {profile: [narrative_text]}
    return narrative_entry

async def generate_narratives_for_model(
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
    model_config = load_yaml("./config/model_config.yaml")
    model_id = model_config[model_name]['model_id']
    model_temp_values = model_config[model_name]['temperature_values']

    for temp_value in model_temp_values:
        for gender_scenario in gender_scenarios:
            logging.info(f"Model ID: {model_id}. Gender scenario: {gender_scenario}. Temperature: {temp_value}.")

            output_file = f"./data/narratives/{model_id}_temp{temp_value}_gender_{gender_scenario}"
            if output_suffix:
                output_file += f"_{output_suffix}"
            output_file += ".json"
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

            prompts = load_json(f'./data/prompts_gender_{gender_scenario}.json')

            # Apply profile slicing, in case we're filling in narratives
            first_idx = first_id-1 # profiles are 1-indexed
            last_idx = last_id if last_id is not None else len(all_profiles) # slicing is exclusive on the upper bound
            # retrieving relevant profiles
            profiles = all_profiles[first_idx:last_idx] 

            logging.warning(f"Processing profiles from {first_id} to {last_id if last_id is not None else 'end'}.")
            logging.info(f"Generating {samples_per_profile} narrative(s) per profile for {len(profiles)} profiles.")

            for i in tqdm(range(samples_per_profile), desc=f"Generating narratives, round {i+1}."):
                logging.info(f"Starting narrative generation, round {i+1}.")
                start = time.time()

                # Create async tasks for profiles
                tasks = []
                for profile in tqdm(profiles):
                    try:
                        prompt = next(
                            filter(lambda item: item['profile_id'] == profile, prompts)
                        )['prompt_text']

                        tasks.append(generate_narrative_for_profile(profile, model_name, prompt, temp_value))
                    except StopIteration:
                        logging.error(f"No prompt found for profile '{profile}'. Skipping.")
                    except Exception as e:
                        logging.error(f"Error generating narrative for profile '{profile}': {e}")

                    results = await asyncio.gather(*tasks)

                for narrative in results:
                    save_json(narrative, output_file)

                end = time.time()
                logging.info(f"Narrative generation round {i+1} completed in {(end - start)/3600:.2f} hours.")


async def generate_narratives_async(model_names, gender_scenarios, samples_per_profile, **kwargs):
    """Runs multiple models asynchronously."""
    logging.info("Starting async narrative generation for multiple models.")
    
    tasks = [generate_narratives_for_model(model_name, gender_scenarios, samples_per_profile, **kwargs)
             for model_name in model_names]

    await asyncio.gather(*tasks)  # Run all model generations in parallel


def generate_narratives(model_names, gender_scenarios, samples_per_profile, **kwargs):
    """Runs a single model synchronously, or multiple models asynchronously."""
    if len(model_names) == 1:
        logging.info(f"Running narrative generation synchronously for model: {model_names[0]}")
        generate_narratives_for_model(model_names[0], gender_scenarios, samples_per_profile, **kwargs)
    else:
        logging.info(f"Running narrative generation asynchronously for models: {', '.join(model_names)}")
        asyncio.run(generate_narratives_async(model_names, gender_scenarios, samples_per_profile, **kwargs))


if __name__ == "__main__":
    configure_logging(script_name='generate_narratives')

    config = load_yaml("./config/narrative_config.yaml")

    model_names = [model for model in config['models'] if config['models'][model]]
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
