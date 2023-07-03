"""Evaluate RPMNet. Also contains functionality to compute evaluation metrics given transforms

Example Usages:
    1. Evaluate RPMNet
        python eval.py --noise_type crop --resume [path-to-model.pth]

    2. Evaluate precomputed transforms (.npy file containing np.array of size (B, 3, 4) or (B, n_iter, 3, 4))
        python eval.py --noise_type crop --transform_file [path-to-transforms.npy]
"""
import os
import torch
from arguments import rpmnet_eval_arguments
from utils import test
from common.misc import prepare_logger


if __name__ == '__main__':
    # Arguments and logging
    parser = rpmnet_eval_arguments()
    args = parser.parse_args()
    _logger, _log_path = prepare_logger(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    if args.gpu >= 0 and (args.method == 'rpm' or args.method == 'rpmnet'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')
    _device = torch.device('cpu')
    test(args, _device, _log_path)
