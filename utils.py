#!/usr/bin/python3
import json
import torch
from torch.utils.data import WeightedRandomSampler
import pandas as pd
import os, time
import numpy as np


def read_json_from_path(json_path):
    """Takes path and returns json object."""
    with open(json_path, "r") as read_file:
        data = json.load(read_file)
    return data

def get_class_distribution(target_list, num_classes):
    """Takes labels of samples and creates a dict with the counts of every class."""
    if num_classes == 3:
        count_dict = {
            "monotonous": 0,
            "normal": 0,
            "enthusiastic": 0
        }

        for i in target_list:
            if i == 0:
                count_dict['monotonous'] += 1
            elif i == 1:
                count_dict['normal'] += 1
            elif i == 2:
                count_dict['enthusiastic'] += 1
            else:
                print("Check classes.")
    elif num_classes == 2:
        count_dict = {
            "non-enthusiastic": 0,
            "enthusiastic": 0
        }

        for i in target_list:
            if i == 0:
                count_dict['non-enthusiastic'] += 1
            elif i == 1:
                count_dict['enthusiastic'] += 1
            else:
                print("Check classes.")

    return count_dict

def get_input_size(feature_path_list):
    """Takes list of path of different CSV files, each containing different features."""
    features = []
    input_size = 0
    for arg in feature_path_list:
        features.append(pd.read_csv(arg, index_col=0))

    for df in features:
        input_size += len(df.iloc[0,0:-1])
    return input_size

def get_Xy_from_csv(csv_path):
    filenames = pd.read_csv(csv_path)
    X = filenames.iloc[:, 0].to_numpy()
    y = filenames.iloc[:, 1].to_numpy()
    return X,y

def get_weighted_sampler(train_dataset, num_classes):
    """Setting up WeightedRandomSampler. This helps improve learning by weighting samples from underrepresented
    classes stronger."""
    target_list = []
    for sample_batched in train_dataset:
        target_list.append(sample_batched[1])

    target_list = torch.tensor(target_list)
    target_list = target_list[torch.randperm(len(target_list))]

    class_count = [i for i in get_class_distribution(target_list, num_classes).values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)

    class_weights_all = class_weights[target_list.long()]

    weighted_sampler = WeightedRandomSampler(
        weights=class_weights_all,
        num_samples=len(class_weights_all),
        replacement=True
    )
    return weighted_sampler, class_weights

def set_directories(config):
    # Check if dir exists
    if not os.path.exists(config.get('model_output_dir')):
        os.makedirs(config.get('model_output_dir'))
    if not os.path.exists(config.get('results_dir')):
        os.makedirs(config.get('results_dir'))
    if not os.path.exists(config.get('tensorboard_dir')):
        os.makedirs(config.get('tensorboard_dir'))


def get_experiment_name(args):
    config_file = args.JSON_path
    config_name = config_file.split('/')[-1]
    config_name = config_name.split('.')[0]
    timestr = time.strftime("%Y%m%d-%H%M%S")

    return config_name + '_' + str(args.num_classes) + '_' + args.model + '_' + args.fusion + '_' + timestr

def get_concat_data_array(dataset):
    X = []
    y = []
    for i in range(dataset.__len__()):
        sample = dataset.__getitem__(i)
        X.append(sample[0])
        y.append(sample[1])

    return np.array(X).astype('float32'),np.array(y).astype('int64')

def get_featurelist(feature_list, args):
    feature_paths = []

    for feature in feature_list:
        feature_paths.append(feature+ '_' + str(args.num_classes)+'class.csv')

    return feature_paths