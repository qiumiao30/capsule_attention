# -*- coding: utf-8 -*-
# @Time    : 2021/4/25 11:51
# @File    : config.py
import warnings
import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Keras TimeSeries Training')

    # config parameter
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--validation_split', default=0.4, type=float,
                        help='train and test validations')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-b', '--batch_size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning Rate")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--model', type=str, default='cnn_lstm',
                        help='model name')
    parser.add_argument('--data_name', type=str, default='swat',
                        help='dataset name')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help="Which optimizer")
    parser.add_argument('--iterations', type=int, default=100000,
                        help="Number of iterations")
    parser.add_argument('--verbose', type=int, default=1,
                        help="verbose")
    parser.add_argument('--save', default=True, action="store_true",
                        help="save to disk?")
    parser.add_argument('--weight_dir', type=str, default="weights",
                        help="Weight path")
    parser.add_argument('--output_dir', type=str, default="prediction",
                        help="Prediction path")
    parser.add_argument('--log_dir', type=str, default="logs",
                        help="Log path")
    parser.add_argument('--train', default=True, action="store_true",
                        help="Train?")
    parser.add_argument('--seq_len', default=64, type=int,
                        help='series length of the sample')
    parser.add_argument('--loss', default='bce', type=str,
                        help='diffent loss functions')
    parser.add_argument('--test_size', default=0.3, type=float,
                        help='train and test split')
    parser.add_argument('--alpha', default=0.25, type=float,
                        help='focal/mccfocal alpha')
    parser.add_argument('--lambd', default=5, type=float,
                        help='focal/mccfocal lambd')

    args = parser.parse_args()
    return args
