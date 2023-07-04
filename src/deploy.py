import os
import torch
import torch.utils.data
from arguments import rpmnet_train_arguments
from common.misc import prepare_logger
from utils import get_model
from flask import Flask, jsonify, request


# Set up arguments and logging
parser = rpmnet_train_arguments()
args = parser.parse_args()
logger, log_path = prepare_logger(args)
if args.gpu >= 0:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
else:
    device = torch.device('cpu')

model = get_model(args, device, log_path)
model.eval()
