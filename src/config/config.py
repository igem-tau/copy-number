import ast
import configparser
import os


DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(DIR, "config.ini")
FEATURES_SECTION_NAME = "ModelFeatures"


# This should be called after feature importance part knows what features we want
def write_features_config(model_features: dict):
    config = configparser.ConfigParser()

    # Add sections and settings
    config.add_section(FEATURES_SECTION_NAME)
    for k, v in model_features.items():
        config.set(FEATURES_SECTION_NAME, k, v)

    # Write the configuration to the file
    with open(CONFIG_PATH, 'w') as configfile:
        config.write(configfile)


def get_config(file_path=CONFIG_PATH):
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve the original case of keys
    config.read(file_path)
    return config


def get_model_features() -> dict:
    conf = get_config()

    model_features = {}
    if conf.has_section(FEATURES_SECTION_NAME):
        for k, v in conf.items(FEATURES_SECTION_NAME):
            model_features[k] = conf.getboolean(FEATURES_SECTION_NAME, k)

    return model_features


def get_section_features(section_name: str) -> dict:
    conf = get_config()

    features = {}
    if conf.has_section(section_name):
        # special case for one hot encoding returns list
        if section_name == "OneHotEncoding":
            if conf.has_option('OneHotEncoding', 'indices'):
                indices_str = conf.get('OneHotEncoding', 'indices')
                indices_list = ast.literal_eval(indices_str)
                return indices_list

        # walk through section items and fill the dict
        for k, v in conf.items(section_name):
            features[k] = conf.getboolean(section_name, k)

    return features