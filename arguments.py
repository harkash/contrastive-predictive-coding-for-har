import argparse

import os
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for Contrastive Predictive Coding for Human '
                    'Activity Recognition')

    # Data loading parameters
    parser.add_argument('--window', type=int, default=50, help='Window size')
    parser.add_argument('--overlap', type=int, default=25,
                        help='Overlap between consecutive windows')

    # Training settings
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--gpu_id', type=str, default='0')

    # Dataset to train on
    parser.add_argument('--dataset', type=str, default='mobiact',
                        help='Choosing the dataset to perform the training on')

    # Conv encoder
    parser.add_argument('--kernel_size', type=int, default=3,
                        help='Size of the conv filters in the encoder')

    # Future prediction horizon
    parser.add_argument('--num_steps_prediction', type=int, default=28,
                        help='Number of steps in the future to predict')

    # ------------------------------------------------------------
    # Classification parameters
    parser.add_argument('--classifier_lr', type=float, default=5e-4,)
    parser.add_argument('--classifier_batch_size', type=int, default=256)
    parser.add_argument('--saved_model', type=str, default=None,
                        help='Full path of the learned CPC model')
    parser.add_argument('--learning_schedule', type=str, default='last_layer',
                        choices=['last_layer', 'all_layers'],
                        help='last layer freezes the encoder weights but '
                             'all_layers does not.')
    # ------------------------------------------------------------

    # Random seed for reproducibility
    parser.add_argument('--random_seed', type=int, default=42)

    args = parser.parse_args()

    # Setting parameters by the dataset
    if args.dataset == 'mobiact':
        args.input_size = 6
        args.num_classes = 11
        args.root_dir = 'data'
        args.data_file = 'mobiact.mat'

    args.device = torch.device(
        "cuda:" + str(args.gpu_id) if torch.cuda.is_available() else "cpu")

    # Conv padding size
    args.padding = int(args.kernel_size // 2)

    return args

