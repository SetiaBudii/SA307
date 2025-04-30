import yaml
import os

def load_config():
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, '..', 'configs', 'config.yml')
    config_path = os.path.normpath(config_path)

    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# config = load_config()

# print("Training started with:")
# print(f"Batch Size: {config['fine_tune_params']['batch_size']}")
