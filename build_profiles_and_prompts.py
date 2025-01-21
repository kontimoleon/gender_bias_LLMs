import os
import itertools

from utils import load_json, save_json, configure_logging


def build_profiles(
        demographic_dimensions_file: str = './data/demographic_dimensions.json',
        output_file: str = './data/synthetic_profiles.json'
    ):
    profiles = {}

    demographics = load_json(demographic_dimensions_file)
    demo_keys, demo_values = zip(*demographics.items())
    combinations = list(itertools.product(*demo_values))

    for idx, combo in enumerate(combinations):
        profiles[str(idx + 1)] = {demo_keys[i]: combo[i] for i in range(len(demo_keys))}

    save_json(data=profiles, file_path=output_file)


def build_prompt_for_profile_and_template(
      profile_id: str,
      prompt_template_id: str,
      synthetic_profiles: dict,
      prompt_templates: dict,
      gender_scenario: bool = True
    ):
    profile = synthetic_profiles.get(profile_id)
    prompt = prompt_templates.get(prompt_template_id)['template']
    if gender_scenario == 'defined':
      prompt = prompt.replace('[gender]', profile['gender'])
    else:
      prompt += "Feel free to make assumptions on the gender of the individual (male or female)" 
      prompt += "\nExplicitly state the assumed gender in your narrative, starting your response with:"
      prompt += "\n\"Assuming the individual is [fill in the blank with assumed gender]\""
    prompt = prompt.replace('[education]', profile['education'])
    prompt = prompt.replace('[occupation]', profile['occupation'])
    prompt = prompt.replace('[civil_status]', profile['civil_status'])
    return prompt
def build_prompts(
      synthetic_profiles_file: str = './data/synthetic_profiles.json',
      prompt_templates_file: str = './data/prompt_templates.json',
      output_dir: str = './data'
      ):
    synthetic_profiles = load_json(synthetic_profiles_file)
    prompt_templates = load_json(prompt_templates_file)

    profile_ids = list(synthetic_profiles.keys())
    template_ids = list(prompt_templates.keys())

    for gender_scenario in ['defined', 'assumed']:
        all_prompts = {
            template: [
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
                for profile in profile_ids
            ]
            for template in template_ids
        }

        output_file = os.path.join(output_dir, f'prompts_gender_{gender_scenario}.json')
        save_json(data=all_prompts, file_path=output_file)


if __name__ == "__main__":
    configure_logging(script_name='build_profiles_and_prompts')
    # Build synthetic profiles
    build_profiles()
    # Build prompts per template and profile
    build_prompts()
