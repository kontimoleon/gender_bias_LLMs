import os
import itertools
import logging

from tqdm import tqdm
from utils import load_json, save_json, configure_logging


def build_profiles(
        demographic_dimensions_file: str = './data/demographic_dimensions.json',
        output_file: str = './data/synthetic_profiles.json'
    ):
    logging.info("Loading demographic dimensions from %s", demographic_dimensions_file)
    demographics = load_json(demographic_dimensions_file)
    demo_keys, demo_values = zip(*demographics.items())

    logging.info("Generating all combinations of demographic dimensions.")
    combinations = list(itertools.product(*demo_values))
    profiles = {}

    logging.info("Building synthetic profiles.")
    for idx, combo in enumerate(tqdm(combinations, desc="Building Profiles")):
        profiles[str(idx + 1)] = {demo_keys[i]: combo[i] for i in range(len(demo_keys))}

    logging.info("Saving synthetic profiles to %s", output_file)
    save_json(data=profiles, file_path=output_file)
    logging.info("Synthetic profiles saved successfully.")


def build_prompt_for_profile_and_template(
      profile_id: str,
      prompt_template_id: str,
      synthetic_profiles: dict,
      prompt_templates: dict,
      gender_scenario: str = "defined"
    ):
    profile = synthetic_profiles.get(profile_id)
    prompt = prompt_templates.get(prompt_template_id)['template']

    if gender_scenario == 'defined':
        prompt = prompt.replace('[gender]', profile['gender'])
    else:
        prompt += (
            " Feel free to make assumptions on the gender of the individual (male or female)."
            "\nExplicitly state the assumed gender in your narrative, starting your response with:"
            "\n\"Assuming the individual is [fill in the blank with assumed gender]\""
        )

    prompt = prompt.replace('[education]', profile['education'])
    prompt = prompt.replace('[occupation]', profile['occupation'])
    prompt = prompt.replace('[civil_status]', profile['civil_status'])

    return prompt


def build_prompts(
      synthetic_profiles_file: str = './data/synthetic_profiles.json',
      prompt_templates_file: str = './data/prompt_templates.json',
      output_dir: str = './data'
    ):
    logging.info("Loading synthetic profiles from %s", synthetic_profiles_file)
    synthetic_profiles = load_json(synthetic_profiles_file)

    logging.info("Loading prompt templates from %s", prompt_templates_file)
    prompt_templates = load_json(prompt_templates_file)

    profile_ids = list(synthetic_profiles.keys())
    template_ids = list(prompt_templates.keys())

    for gender_scenario in ['defined', 'assumed']:
        logging.info("Building prompts for gender scenario: %s", gender_scenario)

        all_prompts = {}
        for template in tqdm(template_ids, desc=f"Templates ({gender_scenario})"):
            all_prompts[template] = [
                {
                    'profile_id': profile,
                    'prompt_text': build_prompt_for_profile_and_template(
                        profile_id=profile,
                        prompt_template_id=template,
                        synthetic_profiles=synthetic_profiles,
                        prompt_templates=prompt_templates,
                        gender_scenario=gender_scenario
                    )
                }
                for profile in tqdm(profile_ids, desc=f"Profiles for {template}", leave=False)
            ]

        output_file = os.path.join(output_dir, f'prompts_gender_{gender_scenario}.json')
        logging.info("Saving prompts to %s", output_file)
        save_json(data=all_prompts, file_path=output_file)
        logging.info("Prompts for gender scenario '%s' saved successfully.", gender_scenario)


if __name__ == "__main__":
    configure_logging(script_name='build_profiles_and_prompts')
    logging.info("Starting the profiles and prompts generation script.")

    try:
        # Build synthetic profiles
        logging.info("Generating synthetic profiles.")
        build_profiles()

        # Build prompts per template and profile
        logging.info("Generating prompts for all templates and profiles.")
        build_prompts()

        logging.info("Profiles and prompts generation completed successfully.")
    except Exception as e:
        logging.error("An error occurred: %s", e, exc_info=True)
