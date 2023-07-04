from collections import defaultdict
import json
import os
import numpy as np
import open3d as o3  # Need to import before torch
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
from typing import Dict, List
from common.math_torch import se3

from common.math.so3 import dcm2euler
from common.torch import dict_all_to_device, CheckPointManager, TorchDebugger
from data_loader.datasets import get_source, generate_data
import models.rpmnet
import trimesh


def compute_metrics(transform_gt, pred_transforms) -> Dict:
    """Compute metrics required in the paper"""

    def square_distance(src, dst):
        return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)

    with torch.no_grad():
        pred_transforms = pred_transforms
        gt_transforms = transform_gt

        # Euler angles, Individual translation errors (Deep Closest Point convention)
        # TODO Change rotation to torch operations
        r_gt_euler_deg = dcm2euler(gt_transforms[:3, :3], seq="xyz")
        r_pred_euler_deg = dcm2euler(pred_transforms[:3, :3].copy(), seq="xyz")
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

        metrics = {"r_mse": r_mse, "r_mae": r_mae, "t_mse": t_mse, "t_mae": t_mae}

    return metrics


def get_mesh(point_cloud):
    point_cloud.estimate_normals()
    # estimate radius for rolling ball
    distances = point_cloud.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    radius = 1.5 * avg_dist   

    mesh = o3.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            point_cloud,
            o3.utility.DoubleVector([radius, radius * 2]))

    return mesh


def generate_target(source_points, ref_points, pred_transforms, data_batch):
    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(source_points)
    o3.io.write_point_cloud("sync.ply", pcd)

    source = o3.io.read_point_cloud("sync.ply")
    source.remove_non_finite_points()

    pcd = o3.geometry.PointCloud()
    pcd.points = o3.utility.Vector3dVector(ref_points)
    o3.io.write_point_cloud("sync1.ply", pcd)

    target = o3.io.read_point_cloud("sync1.ply")
    target.remove_non_finite_points()

    source = source.voxel_down_sample(voxel_size=0.05)
    target = target.voxel_down_sample(voxel_size=0.05)

    transform = pred_transforms[0].cpu().detach().numpy()[0]
    transform = np.vstack((transform, np.array([0, 0, 0, 1])))

    result = copy.deepcopy(source)
    result.transform(transform)
    gt_transforms = data_batch["transform_gt"]
    transform_gt = gt_transforms
    transform_gt = np.vstack((transform_gt, np.array([0, 0, 0, 1])))
    result_gt = copy.deepcopy(source)
    result_gt.transform(transform_gt)

    reg_p2p = o3.pipelines.registration.registration_icp(
        source,
        target,
        0.02,
        transform,
        o3.pipelines.registration.TransformationEstimationPointToPoint(),
        o3.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    result_rpm_icp = copy.deepcopy(source)
    result_rpm_icp.transform(reg_p2p.transformation)
    # draw_registration_result(source, target, reg_p2p.transformation)
    print(data_batch["category"][0])
    rpmnet = compute_metrics(transform_gt, transform)
    rpmnet_icp = compute_metrics(transform_gt, reg_p2p.transformation)

    # average distance between point in result and result_gt
    predicted_points = np.asarray(result.points)
    predicted_icp_points = np.asarray(result_rpm_icp.points)
    gt_points = np.asarray(result_gt.points)

    dist = np.linalg.norm(predicted_points - gt_points, axis=1)
    dist_icp = np.linalg.norm(predicted_icp_points - gt_points, axis=1)
    avg_dist = np.mean(dist)
    max_dist = np.max(dist)
    min_dist = np.min(dist)
    # print("Average distance between point in result and result_gt : ", avg_dist)
    # print("Max distance between point in result and result_gt : ", max_dist)
    # print("Min distance between point in result and result_gt : ", min_dist)

    avg_dist_icp = np.mean(dist_icp)
    max_dist_icp = np.max(dist_icp)
    min_dist_icp = np.min(dist_icp)
    # print("Average distance between point in result and result_gt : ", avg_dist_icp)
    # print("Max distance between point in result and result_gt : ", max_dist_icp)
    # print("Min distance between point in result and result_gt : ", min_dist_icp)

    print("RPMNet : ", rpmnet)
    print("RPMNet_ICP : ", rpmnet_icp)

    return result, source, target, result_gt, result_rpm_icp, rpmnet, rpmnet_icp


def inference(model: torch.nn.Module, args):
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
    endpoints_out = defaultdict(list)

    vis1 = o3.visualization.Visualizer()
    vis1.create_window(window_name="RPMNet", width=960, height=540, left=0, top=0)

    vis2 = o3.visualization.Visualizer()
    vis2.create_window(window_name="RPMNet_ICP", width=960, height=540, left=0, top=600)

    rpm_stats = []
    rpm_icp_stats = []

    source = get_source(args.object_file)
    source_global = copy.deepcopy(source)

    for i in range(args.iterations):
        source = copy.deepcopy(source_global)
        data_batch = generate_data(source, args)

        with torch.no_grad():
            pred_transforms, endpoints = model(data_batch, args.num_reg_iter)

        source_points = data_batch["points_src"][..., :3].cpu().detach().numpy()
        ref_points = data_batch["points_ref"][..., :3].cpu().detach().numpy()

        source_points = np.reshape(
            source_points,
            (source_points.shape[0] * source_points.shape[1], source_points.shape[2]),
        )
        ref_points = np.reshape(
            ref_points, (ref_points.shape[0] * ref_points.shape[1], ref_points.shape[2])
        )

        (
            result,
            source,
            target,
            result_gt,
            result_rpm_icp,
            rpmnet,
            rpmnet_icp,
        ) = generate_target(source_points, ref_points, pred_transforms, data_batch)

        rpm_stats.append({data_batch["category"][0]: rpmnet})
        rpm_icp_stats.append({data_batch["category"][0]: rpmnet_icp})

        draw_registration_result(
            source, target, result, result_gt, result_rpm_icp, vis1, vis2, args.mesh
        )

    results = {"RPMNet": rpm_stats, "RPMNet_ICP": rpm_icp_stats}
    with open("results/results_noisy_partial_0.8_t_4_updated.json", "w") as fp:
        json.dump(results, fp)

    return pred_transforms_all, endpoints_out


def draw_registration_result(
    source, target, result, result_gt, result_rpm_icp, vis1, vis2, mesh=False
):

    source.paint_uniform_color([1, 0, 0])
    target.paint_uniform_color([0, 1, 0])
    result_gt.paint_uniform_color([0, 0, 1])
    result.paint_uniform_color([0, 1, 1])
    result_rpm_icp.paint_uniform_color([0, 1, 1])

    if mesh:
        result = get_mesh(result)
        result_gt = get_mesh(result_gt)
        result_rpm_icp = get_mesh(result_rpm_icp)
        source = get_mesh(source)
        target = get_mesh(target)

    vis1.add_geometry(result)
    # vis1.add_geometry(result_gt)
    vis1.add_geometry(source)
    vis1.add_geometry(target)

    vis2.add_geometry(result_rpm_icp)
    # vis2.add_geometry(result_gt)
    vis2.add_geometry(source)
    vis2.add_geometry(target)

    count = 0
    while count < 1000:
        vis1.update_geometry(result)
        # vis1.update_geometry(result_gt)
        vis1.update_geometry(source)
        vis1.update_geometry(target)

        if not vis1.poll_events():
            break
        vis1.update_renderer()

        vis2.update_geometry(result)
        # vis2.update_geometry(result_gt)
        vis2.update_geometry(source)
        vis2.update_geometry(target)
        if not vis2.poll_events():
            break
        vis2.update_renderer()

        count += 1

    vis1.clear_geometries()
    vis2.clear_geometries()


def get_model(args, device, log_path):
    model = models.rpmnet.get_model(args)
    model.to(device)
    saver = CheckPointManager(os.path.join(log_path, "ckpt", "models"))
    saver.load(args.resume, model)

    return model


def test(args, device, log_path):
    model = get_model(args, device, log_path)
    inference(model, args)


def compute_losses(
    data: Dict,
    pred_transforms: List,
    args,
    endpoints: Dict,
    loss_type: str = "mae",
    reduction: str = "mean",
) -> Dict:
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
    data["transform_gt"] = np.expand_dims(data["transform_gt"], axis=0)
    data["transform_gt"] = torch.from_numpy(data["transform_gt"]).float()

    gt_src_transformed = se3.transform(
        data["transform_gt"], data["points_src"][..., :3]
    )
    if loss_type == "mse":
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.MSELoss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = se3.transform(
                pred_transforms[i], data["points_src"][..., :3]
            )
            if reduction.lower() == "mean":
                losses["mse_{}".format(i)] = criterion(
                    pred_src_transformed, gt_src_transformed
                )
            elif reduction.lower() == "none":
                losses["mse_{}".format(i)] = torch.mean(
                    criterion(pred_src_transformed, gt_src_transformed), dim=[-1, -2]
                )
    elif loss_type == "mae":
        # MSE loss to the groundtruth (does not take into account possible symmetries)
        criterion = nn.L1Loss(reduction=reduction)
        for i in range(num_iter):
            pred_src_transformed = se3.transform(
                pred_transforms[i], data["points_src"][..., :3]
            )
            if reduction.lower() == "mean":
                losses["mae_{}".format(i)] = criterion(
                    pred_src_transformed, gt_src_transformed
                )
            elif reduction.lower() == "none":
                losses["mae_{}".format(i)] = torch.mean(
                    criterion(pred_src_transformed, gt_src_transformed), dim=[-1, -2]
                )
    else:
        raise NotImplementedError

    # Penalize outliers
    for i in range(num_iter):
        ref_outliers_strength = (
            1.0 - torch.sum(endpoints["perm_matrices"][i], dim=1)
        ) * args.wt_inliers
        src_outliers_strength = (
            1.0 - torch.sum(endpoints["perm_matrices"][i], dim=2)
        ) * args.wt_inliers
        if reduction.lower() == "mean":
            losses["outlier_{}".format(i)] = torch.mean(
                ref_outliers_strength
            ) + torch.mean(src_outliers_strength)
        elif reduction.lower() == "none":
            losses["outlier_{}".format(i)] = torch.mean(
                ref_outliers_strength, dim=1
            ) + torch.mean(src_outliers_strength, dim=1)

    discount_factor = 0.5  # Early iterations will be discounted
    total_losses = []
    for k in losses:
        discount = discount_factor ** (num_iter - int(k[k.rfind("_") + 1 :]) - 1)
        total_losses.append(losses[k] * discount)
    losses["total"] = torch.sum(torch.stack(total_losses), dim=0)

    return losses


def train(args, device, logger, log_path):
    """Main train/val loop"""

    logger.debug("Trainer (PID=%d), %s", os.getpid(), args)

    model = get_model(args, device, log_path)
    model.to(device)
    global_step = 0

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    saver = CheckPointManager(
        os.path.join(log_path, "ckpt", "model"), keep_checkpoint_every_n_hours=0.5
    )
    if args.resume is not None:
        global_step = saver.load(args.resume, model, optimizer)

    # trainings
    torch.autograd.set_detect_anomaly(args.debug)
    model.train()

    source = get_source(args.object_file)

    for epoch in range(0, args.epochs):
        tbar = tqdm(total=args.iterations, ncols=args.iterations)
        for i in range(args.iterations):
            train_data = generate_data(source, args)

            global_step += 1

            optimizer.zero_grad()

            # Forward through neural network
            dict_all_to_device(train_data, device)
            pred_transforms, endpoints = model(
                train_data, args.num_train_reg_iter
            )  # Use less iter during training

            # Compute loss, and optimize
            train_losses = compute_losses(
                train_data,
                pred_transforms,
                args,
                endpoints,
                loss_type=args.loss_type,
                reduction="mean",
            )
            if args.debug:
                with TorchDebugger():
                    train_losses["total"].backward()
            else:
                train_losses["total"].backward()
            optimizer.step()

            tbar.set_description("Loss:{:.3g}".format(train_losses["total"]))
            tbar.update(1)

            # if global_step % args.validate_every == 0:  # Validation loop. Also saves checkpoints
            model.eval()
            # val_score = validate(val_loader, model, val_writer, global_step)
            saver.save(model, optimizer, step=global_step, score=0)
            model.train()

        tbar.close()
