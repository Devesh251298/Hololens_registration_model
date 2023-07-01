"""Evaluate RPMNet. Also contains functionality to compute evaluation metrics given transforms

Example Usages:
    1. Evaluate RPMNet
        python eval.py --noise_type crop --resume [path-to-model.pth]

    2. Evaluate precomputed transforms (.npy file containing np.array of size (B, 3, 4) or (B, n_iter, 3, 4))
        python eval.py --noise_type crop --transform_file [path-to-transforms.npy]
"""
from collections import defaultdict
import json
import os
import pickle
import time
from typing import Dict, List
import random

import numpy as np
import open3d as o3 # Need to import before torch
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import torch
import copy

from arguments import rpmnet_eval_arguments
from common.misc import prepare_logger
from common.torch import dict_all_to_device, CheckPointManager, to_numpy
from common.math import se3
from common.math_torch import se3
from common.math.so3 import dcm2euler
from data_loader.datasets import get_test_datasets
import models.rpmnet
import torchvision

import data_loader.transforms as Transforms


def compute_metrics(transform_gt, pred_transforms) -> Dict:
    """Compute metrics required in the paper
    """

    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = transform_gt

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:3, :3], seq='xyz')
        r_pred_euler_deg = dcm2euler(pred_transforms[:3, :3].copy(), seq='xyz')
        t_gt = gt_transforms[:3, 3]
        t_pred = pred_transforms[:3, 3].copy()
        # print("r_gt_euler_deg: ", r_gt_euler_deg)
        # print("r_pred_euler_deg: ", r_pred_euler_deg)

        print("t_gt: ", t_gt)
        print("t_pred: ", t_pred)

        r_mse = np.mean((r_gt_euler_deg - r_pred_euler_deg) ** 2, axis=1)[0]
        r_mae = np.mean(np.abs(r_gt_euler_deg - r_pred_euler_deg), axis=1)[0]
        t_mse = np.mean((t_gt - t_pred) ** 2)
        t_mae = np.mean(np.abs(t_gt - t_pred))



        metrics = {
            'r_mse': r_mse,
            'r_mae': r_mae,
            't_mse': t_mse,
            't_mae': t_mae
        }

    return metrics


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k]**2))
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def print_metrics(logger, summary_metrics: Dict, losses_by_iteration: List = None,
                  title: str = 'Metrics'):
    """Prints out formated metrics to logger"""


    if losses_by_iteration is not None:
        losses_all_str = ' | '.join(['{:.5f}'.format(c) for c in losses_by_iteration])


def inference(data_loader, model: torch.nn.Module):
    """Runs inference over entire dataset

    Args:
        data_loader (torch.utils.data.DataLoader): Dataset loader
        model (model.nn.Module): Network model to evaluate

    Returns:
        pred_transforms_all: predicted transforms (B, n_iter, 3, 4) where B is total number of instances
        endpoints_out (Dict): Network endpoints
    """

    model.eval()

    pred_transforms_all = []
    all_betas, all_alphas = [], []
    total_time = 0.0
    endpoints_out = defaultdict(list)
    total_rotation = []

    vis1 = o3.visualization.Visualizer()
    vis1.create_window(window_name='RPMNet', width=960, height=540, left=0, top=0)

    vis2 = o3.visualization.Visualizer()
    vis2.create_window(window_name='RPMNet_ICP', width=960, height=540, left=0, top=600)


    rpm_stats = []
    rpm_icp_stats = []

    mesh = o3.io.read_triangle_mesh('STL/Segmentation.stl') 
    # Extract the vertex positions
    vertices = np.asarray(mesh.vertices)

    # Scale down the vertices
    scale_factor = 0.01 
    scaled_vertices = vertices * scale_factor

    # Update the mesh with the scaled vertices
    mesh.vertices = o3.utility.Vector3dVector(scaled_vertices)
    vertices = np.asarray(mesh.vertices)

    # Compute the centroid of the mesh vertices
    centroid = np.mean(vertices, axis=0)

    # Translate the vertices to center them around the origin
    centered_vertices = vertices - centroid

    # Update the mesh with the centered vertices
    mesh.vertices = o3.utility.Vector3dVector(centered_vertices)

    # Compute the surface normals
    mesh.compute_vertex_normals()

    pcd = o3.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    pcd = pcd.uniform_down_sample(every_k_points=300)

    source = copy.deepcopy(pcd)
    source.remove_non_finite_points()

    source_points = np.asarray(source.points)
    source_points = source_points - source_points.min(axis = 0)
    source_points = source_points / source_points.max(axis = 0)
    source_points = source_points * 2
    source.points = o3.utility.Vector3dVector(source_points)

    source_points = np.asarray(source.points)
    source_points = source_points - source_points.mean(axis = 0)
    source.points = o3.utility.Vector3dVector(source_points)

    source.estimate_normals()

    source_global = copy.deepcopy(source)

    for i in range(len(data_loader)):
        data_batch = next(iter(data_loader))
        # if data_batch['category'][0] != 'person':=
        #     continue

        source = copy.deepcopy(source_global)

        rot_mag = np.random.uniform(0, 45)
        trans_mag =  np.random.uniform(0, 2)
        num_points = 4000
        partial_p_keep = [1, np.random.uniform(0.5, 1)]
        sample = {'points': np.concatenate((np.asarray(source.points), np.asarray(source.normals)), axis=1), 'label': 'Actual', 'idx': 4, 'category': 'person'}

        transforms =  torchvision.transforms.Compose([Transforms.SetDeterministic(),
                                    Transforms.SplitSourceRef(),
                                    Transforms.RandomCrop(partial_p_keep),
                                    Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                                    Transforms.Resampler(num_points),
                                    Transforms.RandomJitter(),
                                    Transforms.ShufflePoints()])


        # # pred_transforms = pred_transforms
        data_batch = transforms(sample)

        ## convert to torch tensors

        data_batch['points_src'] = torch.from_numpy(data_batch['points_src']).float().cpu()
        data_batch['points_ref'] = torch.from_numpy(data_batch['points_ref']).float().cpu()

        data_batch['points_src'] = data_batch['points_src'].unsqueeze(0)
        data_batch['points_ref'] = data_batch['points_ref'].unsqueeze(0)

        # with torch.no_grad():
        #     pred_transforms, endpoints = model(data_batch, _args.num_reg_iter)

        with torch.no_grad():
            pred_transforms, endpoints = model(data_batch, _args.num_reg_iter)

        source_points = data_batch['points_src'][..., :3].cpu().detach().numpy()
        ref_points = data_batch['points_ref'][..., :3].cpu().detach().numpy()

        source_points = np.reshape(source_points, (source_points.shape[0]*source_points.shape[1],source_points.shape[2]))
        ref_points = np.reshape(ref_points, (ref_points.shape[0]*ref_points.shape[1],ref_points.shape[2]))

        pcd = o3.geometry.PointCloud()
        pcd.points = o3.utility.Vector3dVector(source_points)
        o3.io.write_point_cloud("sync.ply", pcd)

        source = o3.io.read_point_cloud("sync.ply")
        source.remove_non_finite_points()

        pcd = o3.geometry.PointCloud()
        pcd.points = o3.utility.Vector3dVector(ref_points)
        o3.io.write_point_cloud("sync1.ply", pcd)

        ref = o3.io.read_point_cloud("sync1.ply")
        ref.remove_non_finite_points()

        source = source.voxel_down_sample(voxel_size=0.05)
        target = ref.voxel_down_sample(voxel_size=0.05)

        transform = pred_transforms[0].cpu().detach().numpy()[0]
        transform = np.vstack((transform, np.array([0, 0, 0, 1])))
 
        result = copy.deepcopy(source)
        result.transform(transform)
        gt_transforms = data_batch['transform_gt']
        transform_gt = gt_transforms
        transform_gt = np.vstack((transform_gt, np.array([0, 0, 0, 1])))
        result_gt = copy.deepcopy(source)
        result_gt.transform(transform_gt)

        reg_p2p = o3.pipelines.registration.registration_icp(
            source, target, 0.02, transform,
            o3.pipelines.registration.TransformationEstimationPointToPoint(),
            o3.pipelines.registration.ICPConvergenceCriteria(max_iteration = 200))

        result_rpm_icp = copy.deepcopy(source)
        result_rpm_icp.transform(reg_p2p.transformation)
        # draw_registration_result(source, target, reg_p2p.transformation)
        print(data_batch['category'][0])
        rpmnet = compute_metrics(transform_gt, transform)
        rpmnet_icp = compute_metrics(transform_gt, reg_p2p.transformation)

        print("RPMNet : ",rpmnet)
        print("RPMNet_ICP : ",rpmnet_icp)

        rpm_stats.append({data_batch['category'][0] : rpmnet})
        rpm_icp_stats.append({data_batch['category'][0] : rpmnet_icp})

        source.paint_uniform_color([1, 0, 0])
        target.paint_uniform_color([0, 1, 0])
        result_gt.paint_uniform_color([0, 0, 1])
        result.paint_uniform_color([0, 1, 1])
        result_rpm_icp.paint_uniform_color([0, 1, 1])

        vis1.add_geometry(result)
        vis1.add_geometry(result_gt)
        vis1.add_geometry(source)
        vis1.add_geometry(target)

        vis2.add_geometry(result_rpm_icp)
        vis2.add_geometry(result_gt)
        vis2.add_geometry(source)
        vis2.add_geometry(target)
        
        count = 0
        while count<1000:
            vis1.update_geometry(result)
            vis1.update_geometry(result_gt)
            vis1.update_geometry(source)
            vis1.update_geometry(target)
            if not vis1.poll_events():
                break
            vis1.update_renderer()


            vis2.update_geometry(result)
            vis2.update_geometry(result_gt)
            vis2.update_geometry(source)
            vis2.update_geometry(target)
            if not vis2.poll_events():
                break
            vis2.update_renderer()

            count += 1

        vis1.clear_geometries()
        vis2.clear_geometries()

    results = {'RPMNet': rpm_stats, 'RPMNet_ICP': rpm_icp_stats}
    with open('results/results_noisy_partial_0.8_t_4_updated.json', 'w') as fp:
        json.dump(results, fp)

    return pred_transforms_all, endpoints_out

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3.visualization.draw_geometries([source_temp, target_temp])

def evaluate(pred_transforms, data_loader: torch.utils.data.dataloader.DataLoader):
    """ Evaluates the computed transforms against the groundtruth

    Args:
        pred_transforms: Predicted transforms (B, [iter], 3/4, 4)
        data_loader: Loader for dataset.

    Returns:
        Computed metrics (List of dicts), and summary metrics (only for last iter)
    """

    num_processed, num_total = 0, len(pred_transforms)

    if pred_transforms.ndim == 4:
        pred_transforms = torch.from_numpy(pred_transforms).to(_device)
    else:
        assert pred_transforms.ndim == 3 and \
               (pred_transforms.shape[1:] == (4, 4) or pred_transforms.shape[1:] == (3, 4))
        pred_transforms = torch.from_numpy(pred_transforms[:, None, :, :]).to(_device)

    metrics_for_iter = [defaultdict(list) for _ in range(pred_transforms.shape[1])]

    for data in tqdm(data_loader, leave=False):
        dict_all_to_device(data, _device)

        batch_size = 0
        for i_iter in range(pred_transforms.shape[1]):
            batch_size = data['points_src'].shape[0]

            cur_pred_transforms = pred_transforms[num_processed:num_processed+batch_size, i_iter, :, :]
            metrics = compute_metrics(data, cur_pred_transforms)
            for k in metrics:
                metrics_for_iter[i_iter][k].append(metrics[k])
        num_processed += batch_size

    for i_iter in range(len(metrics_for_iter)):
        metrics_for_iter[i_iter] = {k: np.concatenate(metrics_for_iter[i_iter][k], axis=0)
                                    for k in metrics_for_iter[i_iter]}
        summary_metrics = summarize_metrics(metrics_for_iter[i_iter])
        print_metrics(_logger, summary_metrics, title='Evaluation result (iter {})'.format(i_iter))

    return metrics_for_iter, summary_metrics


def save_eval_data(pred_transforms, endpoints, metrics, summary_metrics, save_path):
    """Saves out the computed transforms
    """

    # Save transforms
    np.save(os.path.join(save_path, 'pred_transforms.npy'), pred_transforms)

    # Save endpoints if any
    for k in endpoints:
        if isinstance(endpoints[k], np.ndarray):
            np.save(os.path.join(save_path, '{}.npy'.format(k)), endpoints[k])
        else:
            with open(os.path.join(save_path, '{}.pickle'.format(k)), 'wb') as fid:
                pickle.dump(endpoints[k], fid)

    # Save metrics: Write each iteration to a different worksheet.
    writer = pd.ExcelWriter(os.path.join(save_path, 'metrics.xlsx'))
    for i_iter in range(len(metrics)):
        metrics[i_iter]['r_rmse'] = np.sqrt(metrics[i_iter]['r_mse'])
        metrics[i_iter]['t_rmse'] = np.sqrt(metrics[i_iter]['t_mse'])
        metrics[i_iter].pop('r_mse')
        metrics[i_iter].pop('t_mse')
        metrics_df = pd.DataFrame.from_dict(metrics[i_iter])
        metrics_df.to_excel(writer, sheet_name='Iter_{}'.format(i_iter+1))
    writer.close()

    # Save summary metrics
    summary_metrics_float = {k: float(summary_metrics[k]) for k in summary_metrics}
    with open(os.path.join(save_path, 'summary_metrics.json'), 'w') as json_out:
        json.dump(summary_metrics_float, json_out)



def get_model():
    if _args.method == 'rpmnet':
        assert _args.resume is not None
        model = models.rpmnet.get_model(_args)
        model.to(_device)
        saver = CheckPointManager(os.path.join(_log_path, 'ckpt', 'models'))
        saver.load(_args.resume, model)
    else:
        raise NotImplementedError
    return model


def main():
    # Load data_loader
    test_dataset = get_test_datasets(_args)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=1, shuffle=True)

    if _args.transform_file is not None:
        pred_transforms = np.load(_args.transform_file)
        endpoints = {}
    else:
        model = get_model()
        pred_transforms, endpoints = inference(test_loader, model)  # Feedforward transforms

    # Compute evaluation matrices
    # eval_metrics, summary_metrics = evaluate(pred_transforms, data_loader=test_loader)

    # save_eval_data(pred_transforms, endpoints, eval_metrics, summary_metrics, _args.eval_save_path)
    # _logger.info('Finished')


if __name__ == '__main__':
    # Arguments and logging
    parser = rpmnet_eval_arguments()
    _args = parser.parse_args()
    _logger, _log_path = prepare_logger(_args, log_path=_args.eval_save_path)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
    if _args.gpu >= 0 and (_args.method == 'rpm' or _args.method == 'rpmnet'):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(_args.gpu)
        _device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    else:
        _device = torch.device('cpu')
    _device = torch.device('cpu')
    main()

