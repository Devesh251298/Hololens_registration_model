"""  Train RPMNet
"""
import os
import torch
import torch.utils.data
from arguments import rpmnet_train_arguments
from common.misc import prepare_logger
from utils import train


if __name__ == "__main__":
    # Set up arguments and logging
    parser = rpmnet_train_arguments()
    args = parser.parse_args()
    logger, log_path = prepare_logger(args)
    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        )
    else:
        device = torch.device("cpu")
    train(args, device, logger, log_path)
