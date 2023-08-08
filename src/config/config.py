'''
The idea is to have some kind of dictionary of all the features that we calculate (features_dictionary.ini),
divided into sections according to the types of the features (i.e. 'pssm' would be under the features that are derived
from our database and 'promoter_strength' would be of the features derived from calculation over the whole promoter sequence
as one unit, etc.)
After running the first model and applying "feature selection" the selected features should be written in this dictionary,
then in the pre_process part the feature_select_flag should be set to True when running the functions to create the new
sequences (permutations) with their features
'''


import ast
import configparser
import os

DIR = os.path.dirname(os.path.abspath(__file__))
# CONFIG_PATH = os.path.join(DIR, "config.ini")
CONFIG_PATH = os.path.join(DIR, "features_dictionary.ini")
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
    config = configparser.ConfigParser(allow_no_value=True)  # added the option for no value since the dictionary
    # contains only the names of the features that will be calculated without any value assigned
    config.optionxform = str  # Preserve the original case of keys
    config = configparser.ConfigParser()
    config.read(file_path)
    return config


# def get_model_features() -> dict:
#     conf = get_config()
#
#     model_features = {}
#     if conf.has_section(FEATURES_SECTION_NAME):
#         for k, v in conf.items(FEATURES_SECTION_NAME):
#             model_features[k] = conf.getboolean(FEATURES_SECTION_NAME, k)
#
#     return model_features


# returns all the section names that contain features
def get_nonempty_section_names():
    config = get_config()
    nonempty_sections = []
    for section_name in config.sections():
        if config.options(section_name):
            nonempty_sections.append(section_name)

    return nonempty_sections


# def get_section_features(section_name: str) -> dict:
#     conf = get_config()
#
#     features = {}
#     if conf.has_section(section_name):
#         # special case for one hot encoding returns list
#         if section_name == "OneHotEncoding":
#             if conf.has_option('OneHotEncoding', 'indices'):
#                 indices_str = conf.get('OneHotEncoding', 'indices')
#                 indices_list = ast.literal_eval(indices_str)
#                 return indices_list
#
#         # walk through section items and fill the dict
#         for k, v in conf.items(section_name):
#             features[k] = conf.getboolean(section_name, k)
#
#     return features

# function to extract the features' names as a list from a specific section in the dictionary
def get_section_features(section_name):
    config = get_config()
    # config = configparser.ConfigParser(allow_no_value=True)
    # config.read(CONFIG_PATH)
    # if section_name in config:
    if config.has_section(section_name):
        section = config[section_name]
        feature_list = list(section.keys())
        return feature_list
    else:
        return []


# extract the features from a section that includes both names of features and lists of features
def get_features_titles_from_section(section_name):
    config = get_config()
    section = config[section_name]

    features_dict = {}
    for option, value in section.items():
        if value is None:  # the feature is only a name
            features_dict[option] = []

        elif '[' in value and ']' in value:  # the feature is a list
            if value != '[]':  # refer only to nonempty lists
                features_dict[option] = value.strip('[]').split(',')

        return features_dict


def get_model_features():
    conf = get_config()

    model_features = {}
    if conf.has_section(FEATURES_SECTION_NAME):
        for k, v in conf.items(FEATURES_SECTION_NAME):
            model_features[k] = conf.getboolean(FEATURES_SECTION_NAME, kk)

    return model_features

