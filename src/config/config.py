import configparser


CONFIG_PATH = "config.ini"
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
    config.read(file_path)
    return config


def get_model_features():
    conf = get_config()

    model_features = {}
    if conf.has_section(FEATURES_SECTION_NAME):
        for k, v in conf.items(FEATURES_SECTION_NAME):
            model_features[k] = conf.getboolean(FEATURES_SECTION_NAME, kk)

    return model_features