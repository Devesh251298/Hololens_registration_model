"""Data loader
"""
import logging

import copy
import numpy as np
import open3d as o3d
import torchvision
import torch

import data_loader.transforms as Transforms

_logger = logging.getLogger()


def get_source(filename: str, type: str = "source", args=None):
    mesh = o3d.io.read_triangle_mesh(filename)
    # Extract the vertex positions
    vertices = np.asarray(mesh.vertices)

    # if type == "target":
    #     vertices = vertices * 1000

    # Scale down the vertices
    scale_factor = 0.001
    scaled_vertices = vertices * scale_factor

    # Update the mesh with the scaled vertices
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # vertices = np.asarray(mesh.vertices)

    # # Compute the centroid of the mesh vertices
    # centroid = np.mean(vertices, axis=0)

    # # Translate the vertices to center them around the origin
    # centered_vertices = vertices - centroid

    # # Update the mesh with the centered vertices
    # mesh.vertices = o3d.utility.Vector3dVector(centered_vertices)

    # Compute the surface normals
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    if args.simulated:
        pcd = pcd.uniform_down_sample(every_k_points=300)
    else:
        if type == "target":
            pcd = pcd.uniform_down_sample(every_k_points=2)
        else:
            pcd = pcd.uniform_down_sample(every_k_points=2)

    source = copy.deepcopy(pcd)
    if np.asarray(source.colors).shape[0] == 0:
        source.colors = o3d.utility.Vector3dVector(np.random.rand(*np.asarray(source.points).shape))
    source.remove_non_finite_points()

    source_points = np.asarray(source.points)
    # source_points = source_points - source_points.min(axis = 0)
    # source_points = source_points / source_points.max(axis = 0)
    # source_points = source_points * 2
    # source.points = o3d.utility.Vector3dVector(source_points)

    # source_points = np.asarray(source.points)
    # source_points = source_points - source_points.mean(axis = 0)

    source.points = o3d.utility.Vector3dVector(source_points)
    source.estimate_normals()

    return source


def generate_data(source, target, args):
    rot_mag = np.random.uniform(0, args.rot_mag)
    trans_mag = np.random.uniform(0, args.trans_mag)
    num_points = args.num_points
    prob = np.random.uniform(args.partial_min, 1)
    partial_p_keep = [1, np.random.uniform(prob, 1)]
    # print(np.asarray(source.points).shape, np.asarray(source.normals).shape)
    # print(np.asarray(target.points).shape, np.asarray(target.normals).shape)
    sample = {'points': np.concatenate((np.asarray(source.points), np.asarray(source.normals), np.asarray(source.colors)), axis=1), 'label': 'Actual', 'idx': 4, 'category': 'patient'}
    sample2 = {'points': np.concatenate((np.asarray(target.points), np.asarray(target.normals), np.asarray(target.colors)), axis=1), 'label': 'Actual', 'idx': 4, 'category': 'patient'}

    # if args.simulated:
    transforms1 = torchvision.transforms.Compose([
                                                    Transforms.SplitSourceRef(),
                                                    Transforms.RandomCrop([0.6]),
                                                    Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                                                    Transforms.Resampler(num_points),
                                                    Transforms.RandomJitter(),
                                                    Transforms.ShufflePoints(args)
                                                    ])
    # else:
    transforms2 = torchvision.transforms.Compose([
                                                    Transforms.SplitSourceRef(),
                                                    Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                                                    Transforms.Resampler(num_points),
                                                    ])
    if args.simulated:
        data_batch = transforms1(sample)
        data_batch2 = transforms1(sample2)

    else:
        data_batch = transforms2(sample)
        data_batch2 = transforms2(sample2)

    # data_batch = transforms(sample)
    # data_batch2 = transforms(sample2)

    data_batch['points_src'] = torch.from_numpy(data_batch['points_src']).float().cpu()
    data_batch['points_ref'] = torch.from_numpy(data_batch2['points_ref']).float().cpu()

    data_batch['points_src'] = data_batch['points_src'].unsqueeze(0)
    data_batch['points_ref'] = data_batch['points_ref'].unsqueeze(0)

    return data_batch
