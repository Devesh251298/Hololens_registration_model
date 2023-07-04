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


def get_source(filename: str):
    mesh = o3d.io.read_triangle_mesh(filename)
    # Extract the vertex positions
    vertices = np.asarray(mesh.vertices)

    # Scale down the vertices
    scale_factor = 0.01
    scaled_vertices = vertices * scale_factor

    # Update the mesh with the scaled vertices
    mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)
    vertices = np.asarray(mesh.vertices)

    # Compute the centroid of the mesh vertices
    centroid = np.mean(vertices, axis=0)

    # Translate the vertices to center them around the origin
    centered_vertices = vertices - centroid

    # Update the mesh with the centered vertices
    mesh.vertices = o3d.utility.Vector3dVector(centered_vertices)

    # Compute the surface normals
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
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
    source.points = o3d.utility.Vector3dVector(source_points)

    source_points = np.asarray(source.points)
    source_points = source_points - source_points.mean(axis = 0)
    source.points = o3d.utility.Vector3dVector(source_points)

    source.estimate_normals()

    return source


def generate_data(source, args):
    rot_mag = np.random.uniform(0, args.rot_mag)
    trans_mag = np.random.uniform(0, args.trans_mag)
    num_points = args.num_points
    partial_p_keep = [1, np.random.uniform(args.partial_min, 1)]
    sample = {'points': np.concatenate((np.asarray(source.points), np.asarray(source.normals)), axis=1), 'label': 'Actual', 'idx': 4, 'category': 'person'}

    transforms = torchvision.transforms.Compose([Transforms.SetDeterministic(),
                                                Transforms.SplitSourceRef(),
                                                Transforms.RandomCrop(partial_p_keep),
                                                Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                                                Transforms.Resampler(num_points),
                                                Transforms.RandomJitter(),
                                                Transforms.ShufflePoints(args)])

    data_batch = transforms(sample)

    data_batch['points_src'] = torch.from_numpy(data_batch['points_src']).float().cpu()
    data_batch['points_ref'] = torch.from_numpy(data_batch['points_ref']).float().cpu()

    data_batch['points_src'] = data_batch['points_src'].unsqueeze(0)
    data_batch['points_ref'] = data_batch['points_ref'].unsqueeze(0)

    return data_batch
