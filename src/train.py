""" Train RPMNet

Example usage:
    python train.py --noise_type crop
    python train.py --noise_type jitter --train_batch_size 4
"""
from collections import defaultdict
import os
import random
from typing import Dict, List

from matplotlib.pyplot import cm as colormap
import numpy as np
import open3d  # Ensure this is imported before pytorch
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import open3d as o3

from arguments import rpmnet_train_arguments
from common.colors import BLUE, ORANGE
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, TorchDebugger
from common.math_torch import se3
from data_loader.datasets import get_train_datasets, get_source, generate_data
from eval import compute_metrics
from models.rpmnet import get_model
import data_loader.transforms as Transforms
import copy
import torchvision


def compute_losses(data: Dict, pred_transforms: List, endpoints: Dict,
                   loss_type: str = 'mae', reduction: str = 'mean') -> Dict:
    """Compute losses

    Args:
        data: Current mini-batch data
        pred_transforms: Predicted transform, to compute main registration loss
        endpoints: Endpoints for training. For computing outlier penalty
        loss_type: Registration loss type, either 'mae' (Mean absolute error, used in paper) or 'mse'
        reduction: Either 'mean' or 'none'. Use 'none' to accumulate losses outside
                   (useful for accumulating losses for entire validation dataset)

    Returns:
        losses: Dict containing various fields. Total loss to be optimized is in losses['total']

    """

    losses = {}
    num_iter = len(pred_transforms)

    # Compute losses
    # extend dimension to 1 x 4 x 4 which is a numpy array
    data['transform_gt'] = np.expand_dims(data['transform_gt'], axis=0)
    data['transform_gt'] = torch.from_numpy(data['transform_gt']).float()
    
    gt_src_transformed = se3.transform(data['transform_gt'], data['points_src'][..., :3])
    if loss_type == 'mse':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
            if reduction.lower() == 'mean':
                losses['mse_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mse_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    elif loss_type == 'mae':
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = se3.transform(pred_transforms[i], data['points_src'][..., :3])
            if reduction.lower() == 'mean':
                losses['mae_{}'.format(i)] = criterion(pred_src_transformed, gt_src_transformed)
            elif reduction.lower() == 'none':
                losses['mae_{}'.format(i)] = torch.mean(criterion(pred_src_transformed, gt_src_transformed),
                                                        dim=[-1, -2])
    else:
        raise NotImplementedError

    # Penalize outliers
    for i in range(num_iter):
        ref_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=1)) * _args.wt_inliers
        src_outliers_strength = (1.0 - torch.sum(endpoints['perm_matrices'][i], dim=2)) * _args.wt_inliers
        if reduction.lower() == 'mean':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength) + torch.mean(src_outliers_strength)
        elif reduction.lower() == 'none':
            losses['outlier_{}'.format(i)] = torch.mean(ref_outliers_strength, dim=1) + \
                                             torch.mean(src_outliers_strength, dim=1)

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind('_')+1:]) - 1)
        total_losses.append(losses[k] * discount)
    losses['total'] = torch.sum(torch.stack(total_losses), dim=0)

    return losses


def main():
    """Main train/val loop"""

    _logger.debug('Trainer (PID=%d), %s', os.getpid(), _args)

    model = get_model(_args)
    model.to(_device)
    global_step = 0

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=_args.lr)

    saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'model'), keep_checkpoint_every_n_hours=0.5)
    if _args.resume is not None:
        global_step = saver.load(_args.resume, model, optimizer)

    # trainings
    torch.autograd.set_detect_anomaly(_args.debug)
    model.train()

    source = get_source()

    for epoch in range(0, 5):
        tbar = tqdm(total=100, ncols=100)
        for i in range(100):
            train_data = generate_data(source)

            train_data['points_src'] = torch.from_numpy(train_data['points_src']).float().cpu()
            train_data['points_ref'] = torch.from_numpy(train_data['points_ref']).float().cpu()

            train_data['points_src'] = train_data['points_src'].unsqueeze(0)
            train_data['points_ref'] = train_data['points_ref'].unsqueeze(0)
            global_step += 1

            optimizer.zero_grad()

            # Forward through neural network
            dict_all_to_device(train_data, _device)
            pred_transforms, endpoints = model(train_data, _args.num_train_reg_iter)  # Use less iter during training

            # Compute loss, and optimize
            train_losses = compute_losses(train_data, pred_transforms, endpoints,
                                          loss_type=_args.loss_type, reduction='mean')
            if _args.debug:
                with TorchDebugger():
                    train_losses['total'].backward()
            else:
                train_losses['total'].backward()
            optimizer.step()

            tbar.set_description('Loss:{:.3g}'.format(train_losses['total']))
            tbar.update(1)

            # if global_step % _args.validate_every == 0:  # Validation loop. Also saves checkpoints
            model.eval()
            # val_score = validate(val_loader, model, val_writer, global_step)
            saver.save(model, optimizer, step=global_step, score=0)
            model.train()

        tbar.close()


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
    main()
