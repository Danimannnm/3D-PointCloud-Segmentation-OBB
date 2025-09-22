#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to start a training on ModelNet40 dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import signal
import os
import numpy as np
import sys
import time
import argparse
import torch

# Dataset
from datasets.ModelNet40 import *
from datasets.S3DIS import *
from datasets.SensatUrban import *
from datasets.SemanticKitti import *
from datasets.Toronto3D import *
from torch.utils.data import DataLoader

from utils.config import Config
from utils.tester import ModelTester
from models.architectures import KPCNN, KPFCNN


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

def model_choice(chosen_log):

    ###########################
    # Call the test initializer
    ###########################

    # Automatically retrieve the last trained model
    if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS', 'last_sensaturban', 'last_Toronto3D']:

        # Dataset name
        test_dataset = '_'.join(chosen_log.split('_')[1:])

        # List all training logs
        logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])

        # Find the last log of asked dataset
        for log in logs[::-1]:
            log_config = Config()
            log_config.load(log)
            if log_config.dataset.startswith(test_dataset):
                chosen_log = log
                break

        if chosen_log in ['last_ModelNet40', 'last_ShapeNetPart', 'last_S3DIS', 'last_SensatUrban', 'last_Toronto3D']:
            raise ValueError('No log of the dataset "' + test_dataset + '" found')

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    return chosen_log


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ###############################
    # Choose the model to visualize
    ###############################

    #   Here you can choose which model you want to test with the variable test_model. Here are the possible values :
    #
    #       > 'last_XXX': Automatically retrieve the last trained model on dataset XXX
    #       > '(old_)results/Log_YYYY-MM-DD_HH-MM-SS': Directly provide the path of a trained model

    # Ensure Windows-safe multiprocessing
    try:
        torch.multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # CLI args
    parser = argparse.ArgumentParser(description='KPConv test/inference runner')
    parser.add_argument('--log', type=str, default='last_Toronto3D', help='Log dir path or last_XXX alias')
    parser.add_argument('--chkp', type=str, default='current', help="Checkpoint: 'current', integer index, or path to .tar")
    parser.add_argument('--split', type=str, default='validation', choices=['validation', 'test'], help='Dataset split to evaluate')
    parser.add_argument('--workers', type=int, default=0, help='Number of DataLoader workers (Windows-safe <=1 recommended)')
    parser.add_argument('--gpu', type=str, default='0', help='CUDA_VISIBLE_DEVICES id')
    parser.add_argument('--num_votes', type=int, default=20, help='Number of votes/passes during testing smoothing')
    parser.add_argument('--verbose_calib', action='store_true', help='Verbose sampler calibration logging')
    args = parser.parse_args()

    chosen_log = args.log
    on_val = (args.split == 'validation')

    # Deal with 'last_XXXXXX' choices
    chosen_log = model_choice(chosen_log)

    ############################
    # Initialize the environment
    ############################

    # Set which gpu is going to be used
    GPU_ID = args.gpu

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    ###############
    # Previous chkp
    ###############

    # Resolve checkpoint path (supports 'current', integer index, or direct file path)
    chkp_dir = os.path.join(chosen_log, 'checkpoints')
    chosen_chkp = None
    if isinstance(args.chkp, str) and args.chkp.endswith('.tar') and os.path.exists(args.chkp):
        chosen_chkp = args.chkp
    else:
        # Gather available checkpoints
        if not os.path.isdir(chkp_dir):
            raise FileNotFoundError(f'No checkpoints directory found at {chkp_dir}')
        chkps = [f for f in os.listdir(chkp_dir) if f.endswith('.tar') and f.startswith('chkp_')]
        if args.chkp == 'current' or args.chkp is None:
            chosen_chkp = os.path.join(chkp_dir, 'current_chkp.tar')
        else:
            # Try to parse as index
            try:
                idx = int(args.chkp)
                chosen_chkp = os.path.join(chkp_dir, np.sort(chkps)[idx])
            except Exception:
                # Fallback to current
                chosen_chkp = os.path.join(chkp_dir, 'current_chkp.tar')
    if not os.path.exists(chosen_chkp):
        raise FileNotFoundError(f'Checkpoint not found: {chosen_chkp}')

    # Initialize configuration class
    config = Config()
    config.load(chosen_log)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_symmetries = False
    #config.batch_num = 3
    #config.in_radius = 4
    config.validation_size = 200
    # Windows-friendly: keep workers low
    config.input_threads = max(0, min(args.workers, 2))
    # Ensure saving path is consistent for tester outputs
    config.saving = True
    config.saving_path = chosen_log.replace('\\', '/')

    ##############
    # Prepare Data
    ##############

    print()
    print('Data Preparation')
    print('****************')

    if on_val:
        set = 'validation'
    else:
        set = 'test'

    # Initiate dataset
    if config.dataset == 'ModelNet40':
        test_dataset = ModelNet40Dataset(config, train=False)
        test_sampler = ModelNet40Sampler(test_dataset)
        collate_fn = ModelNet40Collate
    elif config.dataset == 'S3DIS':
        test_dataset = S3DISDataset(config, set='validation', use_potentials=True)
        test_sampler = S3DISSampler(test_dataset)
        collate_fn = S3DISCollate
    elif config.dataset == 'SensatUrban':
        test_dataset = SensatUrbanDataset(config, set='validation', use_potentials=True)
        test_sampler = SensatUrbanSampler(test_dataset)
        collate_fn = SensatUrbanCollate
    elif config.dataset == 'Toronto3D':
        test_dataset = Toronto3DDataset(config, set=set, use_potentials=True)
        test_sampler = Toronto3DSampler(test_dataset)
        collate_fn = Toronto3DCollate
    elif config.dataset == 'SemanticKitti':
        test_dataset = SemanticKittiDataset(config, set=set, balance_classes=False)
        test_sampler = SemanticKittiSampler(test_dataset)
        collate_fn = SemanticKittiCollate
    else:
        raise ValueError('Unsupported dataset : ' + config.dataset)

    # Data loader
    pin_mem = torch.cuda.is_available()
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             sampler=test_sampler,
                             collate_fn=collate_fn,
                             num_workers=config.input_threads,
                             pin_memory=pin_mem)

    # Calibrate samplers
    print(f'Calibration: workers={config.input_threads}, split={set}, log={chosen_log}')
    test_sampler.calibration(test_loader, verbose=args.verbose_calib)

    print('\nModel Preparation')
    print('*****************')

    # Define network model
    t1 = time.time()
    if config.dataset_task == 'classification':
        net = KPCNN(config)
    elif config.dataset_task in ['cloud_segmentation', 'slam_segmentation']:
        net = KPFCNN(config, test_dataset.label_values, test_dataset.ignored_labels)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)

    # Define a visualizer class
    tester = ModelTester(net, chkp_path=chosen_chkp)
    print('Done in {:.1f}s\n'.format(time.time() - t1))

    print('\nStart test')
    print('**********\n')

    # Log run context
    print(f'Run context:')
    print(f'  Dataset     : {config.dataset} ({set})')
    print(f'  Log         : {chosen_log}')
    print(f'  Checkpoint  : {chosen_chkp}')
    print(f'  Num workers : {config.input_threads}')
    print(f'  Device      : {"cuda" if torch.cuda.is_available() else "cpu"}')
    print(f'  Saving to   : test/{config.saving_path.split("/")[-1]}')

    # Training
    if config.dataset_task == 'classification':
        tester.classification_test(net, test_loader, config, num_votes=args.num_votes)
    elif config.dataset_task == 'cloud_segmentation':
        tester.cloud_segmentation_test(net, test_loader, config, num_votes=args.num_votes)
    elif config.dataset_task == 'slam_segmentation':
        tester.slam_segmentation_test(net, test_loader, config, num_votes=args.num_votes)
    else:
        raise ValueError('Unsupported dataset_task for testing: ' + config.dataset_task)
