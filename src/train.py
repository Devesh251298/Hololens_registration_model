"""  Train RPMNet
"""
import os
import torch
import torch.utils.data
from arguments import rpmnet_train_arguments
from common.misc import prepare_logger
from utils import train


if __name__ == '__main__':
    # Set up arguments and logging
    parser = rpmnet_train_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args)
    if _args.gpu >= 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')
    train(_args, _device, _logger, _log_path)
