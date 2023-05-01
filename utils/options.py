# -*- encoding: utf-8 -*-

import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tag',
                        '-t',
                        type=str,
                        default='cache',
                        help='folder name to save the outputs')

    parser.add_argument('--batch_size',
                        '-b',
                        type=int,
                        help="input batch size")

    parser.add_argument('--mu',
                        type=float,
                        help="balence parameter of cnn and gcn")

    parser.add_argument('--lambda',
                        type=float,
                        help="balence parameter of mean and max")

    parser.add_argument('--resume_path',
                        '-r',
                        type=str,
                        help="which path to resume model")

    parser.add_argument('--specific_cfg',
                        '-s',
                        type=str,
                        default=None,
                        help="which path to resume model")

    parser.add_argument('--gpu-id', type=int, help="GPU index")

    return parser.parse_args()
