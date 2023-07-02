"""Data loader
"""
import argparse
import logging
import os
from typing import List

import h5py
import copy
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import torchvision

import data_loader.transforms as Transforms
import common.math.se3 as se3

_logger = logging.getLogger()


def get_source():
    mesh = o3d.io.read_triangle_mesh('STL/Segmentation.stl')
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


def generate_data(source):
    rot_mag = np.random.uniform(0, 45)
    trans_mag = np.random.uniform(0, 2)
    num_points = 4000
    partial_p_keep = [1, np.random.uniform(0.5, 1)]
    sample = {'points': np.concatenate((np.asarray(source.points), np.asarray(source.normals)), axis=1), 'label': 'Actual', 'idx': 4, 'category': 'person'}

    transforms = torchvision.transforms.Compose([Transforms.SetDeterministic(),
                                                Transforms.SplitSourceRef(),
                                                Transforms.RandomCrop(partial_p_keep),
                                                Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                                                Transforms.Resampler(num_points),
                                                Transforms.RandomJitter(),
                                                Transforms.ShufflePoints(add_noise=True)])

    data_batch = transforms(sample)

    return data_batch


def get_train_datasets(args: argparse.Namespace):
    train_categories, val_categories = None, None
    if args.train_categoryfile:
        train_categories = [line.rstrip('\n') for line in open(args.train_categoryfile)]
        train_categories.sort()
    if args.val_categoryfile:
        val_categories = [line.rstrip('\n') for line in open(args.val_categoryfile)]
        val_categories.sort()

    train_transforms, val_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                                      args.num_points, args.partial)
    train_transforms = torchvision.transforms.Compose(train_transforms)
    val_transforms = torchvision.transforms.Compose(val_transforms)

    if args.dataset_type == 'modelnet_hdf':
        train_data = ModelNetHdf(args.dataset_path, subset='train', categories=train_categories,
                                 transform=train_transforms)
        val_data = ModelNetHdf(args.dataset_path, subset='test', categories=val_categories,
                               transform=val_transforms)
    else:
        raise NotImplementedError

    return train_data, val_data


def get_test_datasets(args: argparse.Namespace):
    test_categories = None
    if args.test_category_file:
        test_categories = [line.rstrip('\n') for line in open(args.test_category_file)]
        test_categories.sort()

    _, test_transforms = get_transforms(args.noise_type, args.rot_mag, args.trans_mag,
                                        args.num_points, args.partial)
    test_transforms = torchvision.transforms.Compose(test_transforms)

    if args.dataset_type == 'modelnet_hdf':
        test_data = ModelNetHdf(args.dataset_path, subset='test', categories=test_categories,
                                transform=test_transforms)
    else:
        raise NotImplementedError

    return test_data


def get_transforms(noise_type: str,
                   rot_mag: float = 45.0, trans_mag: float = 0.5,
                   num_points: int = 1024, partial_p_keep: List = None):
    """Get the list of transformation to be used for training or evaluating RegNet

    Args:
        noise_type: Either 'clean', 'jitter', 'crop'.
          Depending on the option, some of the subsequent arguments may be ignored.
        rot_mag: Magnitude of rotation perturbation to apply to source, in degrees.
          Default: 45.0 (same as Deep Closest Point)
        trans_mag: Magnitude of translation perturbation to apply to source.
          Default: 0.5 (same as Deep Closest Point)
        num_points: Number of points to uniformly resample to.
          Note that this is with respect to the full point cloud. The number of
          points will be proportionally less if cropped
        partial_p_keep: Proportion to keep during cropping, [src_p, ref_p]
          Default: [0.7, 0.7], i.e. Crop both source and reference to ~70%

    Returns:
        train_transforms, test_transforms: Both contain list of transformations to be applied
    """

    partial_p_keep = partial_p_keep if partial_p_keep is not None else [0.7, 0.7]

    partial_p_keep = [1, 0.8]

    if noise_type == "clean":
        # 1-1 correspondence for each point (resample first before splitting), no noise
        train_transforms = [Transforms.Resampler(num_points),
                            Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.FixedResampler(num_points),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.ShufflePoints()]

    elif noise_type == "jitter":
        # Points randomly sampled (might not have perfect correspondence), gaussian noise to position
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints()]

    elif noise_type == "crop":
        # Both source and reference point clouds cropped, plus same noise in "jitter"
        train_transforms = [Transforms.SplitSourceRef(),
                            Transforms.RandomCrop(partial_p_keep),
                            Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                            Transforms.Resampler(num_points),
                            Transforms.RandomJitter(),
                            Transforms.ShufflePoints()]

        test_transforms = [Transforms.SetDeterministic(),
                           Transforms.SplitSourceRef(),
                           Transforms.RandomCrop(partial_p_keep),
                           Transforms.RandomTransformSE3_euler(rot_mag=rot_mag, trans_mag=trans_mag),
                           Transforms.Resampler(num_points),
                           Transforms.RandomJitter(),
                           Transforms.ShufflePoints(add_noise=True)]
    else:
        raise NotImplementedError

    return train_transforms, test_transforms


class ModelNetHdf(Dataset):
    def __init__(self, dataset_path: str, subset: str = 'train', categories: List = None, transform=None):
        """ModelNet40 dataset from PointNet.
        Automatically downloads the dataset if not available

        Args:
            dataset_path (str): Folder containing processed dataset
            subset (str): Dataset subset, either 'train' or 'test'
            categories (list): Categories to use
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._root = dataset_path

        metadata_fpath = os.path.join(self._root, '{}_files.txt'.format(subset))

        if not os.path.exists(os.path.join(dataset_path)):
            self._download_dataset(dataset_path)

        with open(os.path.join(dataset_path, 'shape_names.txt')) as fid:
            self._classes = [l.strip() for l in fid]
            self._category2idx = {e[1]: e[0] for e in enumerate(self._classes)}
            self._idx2category = self._classes

        with open(os.path.join(dataset_path, '{}_files.txt'.format(subset))) as fid:
            h5_filelist = [line.strip() for line in fid]
            h5_filelist = [x.replace('data/modelnet40_ply_hdf5_2048/', '') for x in h5_filelist]
            h5_filelist = [os.path.join(self._root, f) for f in h5_filelist]

        if categories is not None:
            categories_idx = [self._category2idx[c] for c in categories]
            self._classes = categories
        else:
            categories_idx = None

        self._data, self._labels = self._read_h5_files(h5_filelist, categories_idx)
        # print(self._labels)
        # print(categories_idx)
        # print(self._classes)
        # create a diction of classes and their indices
        self._category2idx = {e[0]: e[1] for e in enumerate(self._classes)}
        # print(self._category2idx)
        # self._data, self._labels = self._data[:32], self._labels[:32, ...]
        self._transform = transform

    def __getitem__(self, item):
        points = self._data[item:item+1, :, :]
        # concat points based on first dimension
        points = np.concatenate(points, axis=0)
        sample = {'points': points, 'label': self._labels[item], 'idx': np.array(item, dtype=np.int32), 'category': self._idx2category[self._labels[item]]}

        if self._transform:
            sample = self._transform(sample)

        return sample

    def __len__(self):
        return self._data.shape[0]

    @property
    def classes(self):
        return self._classes

    @staticmethod
    def _read_h5_files(fnames, categories):

        all_data = []
        all_labels = []

        for fname in fnames:
            f = h5py.File(fname, mode='r')
            data = np.concatenate([f['data'][:], f['normal'][:]], axis=-1)
            labels = f['label'][:].flatten().astype(np.int64)

            if categories is not None:  # Filter out unwanted categories
                mask = np.isin(labels, categories).flatten()
                data = data[mask, ...]
                labels = labels[mask, ...]

            all_data.append(data)
            all_labels.append(labels)

        all_data = np.concatenate(all_data, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return all_data, all_labels

    @staticmethod
    def _download_dataset(dataset_path: str):
        os.makedirs(dataset_path, exist_ok=True)

        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget --no-check-certificate {}'.format(www))
        os.system('unzip {} -d .'.format(zipfile))
        os.system('mv {} {}'.format(zipfile[:-4], os.path.dirname(dataset_path)))
        os.system('rm {}'.format(zipfile))

    def to_category(self, i):
        return self._idx2category[i]
