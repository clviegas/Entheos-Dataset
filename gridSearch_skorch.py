"""
Carla Viegas
GridSearch combined with 10-fold CV using Stratified random splits.

"""
import numpy as np
import pickle
import torch
import utils
import logging
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import torch.nn.functional as F
import argparse, os, time, random
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit
from skorch.callbacks import EpochScoring
from models import NeuralNet, NeuralNetBatchNorm, NeuralNetDropOut
import csv_dataset_skorch
from sklearn.metrics import confusion_matrix, classification_report
from skorch.callbacks import EarlyStopping

# Set all random seeds to obtained deterministic results
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic=True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Check if config file exists
    if not os.path.isfile(args.JSON_path):
        print('This config file does not exist: ', args.JSON_path)
        exit()

    # Create unique experiment name to save results
    experiment_name = utils.get_experiment_name(args)

    # Load JSON file that contains classifiers and parameters
    config = utils.read_json_from_path(args.JSON_path)

    # Set directories
    model_output_path = os.path.join(config.get('model_output_dir'), experiment_name + '.ckpt')
    results_path = os.path.join(config.get('results_dir'), experiment_name + '.txt')
    tensorboard_path = os.path.join(config.get('tensorboard_dir'), experiment_name)

    # Checks if directories exist and if not create them
    utils.set_directories(config)


    # Start Logging
    logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=results_path, level=logging.DEBUG)

    logging.info('Features: ' + ';'.join(str(x) for x in config.get('feature')))
    logging.info('Model: ' + args.model)
    logging.info('Fusion: ' + args.fusion)
    logging.info('Scoring: ' + args.scoring)


    # -----
    # DATA
    # -----
    X, y = utils.get_Xy_from_csv(args.train_split)
    feature_list = utils.get_featurelist(config.get('feature'), args)
    train_dataset = csv_dataset_skorch.Concat_CSVDataset(X, feature_list )
    X, y = utils.get_concat_data_array(train_dataset)
    # ------
    # MODEL
    # ------
    # Get class weights
    class_weights = utils.get_weighted_sampler(train_dataset, args.num_classes)

    # Callbacks
    balanced_accuracy = EpochScoring(name='bal_acc', scoring='balanced_accuracy', lower_is_better=False)
    accuracy = EpochScoring(scoring='accuracy', lower_is_better=False)
    f1_weighted = EpochScoring(scoring='f1_weighted', lower_is_better=False)
    precision = EpochScoring(scoring='precision_weighted', lower_is_better=False)
    recall = EpochScoring(scoring='recall_weighted', lower_is_better=False)
    early_stop = EarlyStopping(patience=25)

    # The Neural Net is instantiated, none hyperparameter is provided
    input_size = utils.get_input_size(feature_list)

    if args.model == 'NeuralNet':
        model = NeuralNetClassifier(NeuralNet, verbose=4, device=device, callbacks=[
            ('bal_acc',balanced_accuracy), ('acc',accuracy), ('f1_weighted', f1_weighted), ('recall', recall),
            ('precision', precision), early_stop], criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam,
                                    criterion__weight=class_weights)
    elif args.model == 'NeuralNetBatchNorm':
        model = NeuralNetClassifier(NeuralNetBatchNorm, verbose=4, device=device, callbacks=[
            ('bal_acc',balanced_accuracy), ('acc',accuracy), ('f1_weighted', f1_weighted), ('recall', recall),
            ('precision', precision)], criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, criterion__weight=class_weights)
    elif args.model == 'NeuralNetDropOut':
        model = NeuralNetClassifier(NeuralNetDropOut, verbose=4, device=device, callbacks=[
            ('bal_acc',balanced_accuracy), ('acc',accuracy), ('f1_weighted', f1_weighted), ('recall', recall),
            ('precision', precision)], criterion=nn.CrossEntropyLoss, optimizer=torch.optim.Adam, criterion__weight=class_weights)


    params = {
        'max_epochs': config.get('max_epoch'),
        'lr': config.get('lr'),
        'batch_size': [50,100],
        'device':['cuda'],
        'module__input_size': [input_size],
        'module__hidden_size': config.get('hidden_size'),
        'module__num_classes': [args.num_classes],
        'optimizer': [torch.optim.Adam],
        }

    # The grid search module is instantiated
    cross_validation = StratifiedShuffleSplit(n_splits=10, random_state=0)
    gs = GridSearchCV(model, params, refit=True, cv=cross_validation, scoring='roc_auc_ovo_weighted', verbose=2,
                      n_jobs=3)




    # -----
    # TRAINING
    # -----

    # Initialize grid search
    gs.fit(X,y)
    print('Best score and best parameters during 10-fold CV')
    print(gs.best_score_, gs.best_params_)
    logging.info('Best parameter set found on development set during 10-fold CV: \n %s \n', gs.best_params_)
    logging.info("Grid scores on development set:\n")

    means = gs.cv_results_['mean_test_score']
    stds = gs.cv_results_['std_test_score']

    for mean, std, params in zip(means, stds, gs.cv_results_['params']):
        logging.info("%0.3f (+/-%0.03f) for %r"
                     % (mean, std, params))
    # -----
    # TESTING
    # -----
    logging.info('=========================')
    X_test, y_test = utils.get_Xy_from_csv(args.test_split)

    test_dataset = csv_dataset_skorch.Concat_CSVDataset(X_test, feature_list)
    X_test, y_test = utils.get_concat_data_array(test_dataset)

    y_hat = gs.predict(X_test)
    cm = confusion_matrix(y_test, y_hat)
    print('Confusion Matrix on TEST set using best model:')
    print(cm)
    logging.info('TESTING on model trained with best parameters\n')
    logging.info('Confusion Matrix:\n')
    logging.info('\n'+ str(cm))

    cr = classification_report(y_test, y_hat)
    logging.info('Classification report:\n')
    logging.info('\n'+ str(cr))
    print('Classification Report on TEST set')
    print(cr)
    # Save
    with open(model_output_path, 'wb') as f:
        pickle.dump(gs.best_estimator_, f)

if __name__=="__main__":
    my_parser = argparse.ArgumentParser(
        description='This script takes as input a JSON file which contains all parameters. Also csv file with '
                    'filenames that are in training set.')
    my_parser.add_argument('--JSON_path', default=False, help='Path to JSON file')
    my_parser.add_argument('--train_split', default='./train.csv',
                           help='CSV file with filenames and labels of training split')
    my_parser.add_argument('--test_split', default='./test.csv', help='CSV file '
                                                                                                          'with filenames and labels of training split')
    my_parser.add_argument('--num_classes', default=3, type=int, help='Number of classes')
    my_parser.add_argument('--model', default='NeuralNet', help='Model name')
    my_parser.add_argument('--fusion', default='concat', help='Fusion method')
    my_parser.add_argument('--scoring', default='balanced_accuracy', help='Scoring function for gridsearch')
    args = my_parser.parse_args()

    main(args)
