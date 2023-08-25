"""Functions for random sampling"""
import numpy as np


def uniform_2_sphere(num: int = None, radius: float = 1.0, center = [0, 0, 0]):
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        num: Number of vectors to sample (or None if single)

    Returns:
        Random Vector (np.ndarray) of size (num, 3) with norm 1.
        If num is None returned value will have size (3,)

    """
    if num is not None:
        phi = np.random.uniform(0.0, 2 * np.pi, num)
        cos_theta = np.random.uniform(-1.0, 1.0, num)
    else:
        phi = np.random.uniform(0.0, 2 * np.pi)
        cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi) * radius
    y = np.sin(theta) * np.sin(phi) * radius
    z = np.cos(theta) * radius

    return np.stack((x+center[0], y+center[1], z+center[2]), axis=-1)


def uniform_2_plane(num: int = None, width: float = 1.0, height: float = 1.0, center=(0, 0)):
    """Uniform sampling on a 2D plane

    Args:
        num: Number of points to sample (or None if single)
        width: Width of the sampling area in x-direction
        height: Height of the sampling area in y-direction
        center: Center of the sampling area (default is (0, 0))

    Returns:
        Random point (np.ndarray) of size (num, 2).
        If num is None, the returned value will have size (2,).

    """
    if num is not None:
        x = np.random.uniform(-width/2, width/2, num)
        y = np.random.uniform(-height/2, height/2, num)
    else:
        x = np.random.uniform(-width/2, width/2)
        y = np.random.uniform(-height/2, height/2)
    
    z = np.zeros_like(x)

    return np.stack((x+center[0], y+center[1], z), axis=-1)

def create_3d_plane(normal_vector, point_on_plane, n, width = 1, height = 1):
    # Normalize the normal vector to ensure its length is 1
    normal_vector = np.array(normal_vector) / np.linalg.norm(normal_vector)
    
    # Generate two orthogonal vectors that lie on the plane
    v1 = np.random.rand(3)
    v1 = v1 - np.dot(v1, normal_vector) * normal_vector
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(normal_vector, v1)
    
    # Generate random points on the plane
    u = np.random.uniform(-width/2, width/2, n)
    v = np.random.uniform(-height/2, height/2, n)
    points_on_plane = point_on_plane + u[:, np.newaxis] * v1 + v[:, np.newaxis] * v2
    # u = np.random.rand(n)
    # v = np.random.rand(n)
    # points_on_plane = point_on_plane + u[:, np.newaxis] * v1 + v[:, np.newaxis] * v2

    point_on_plane = np.array(point_on_plane)
    
    return points_on_plane

if __name__ == '__main__':
    # Visualize sampling
    from vtk_visualizer.plot3d import plotxyz
    rand_2s = uniform_2_sphere(10000)
    plotxyz(rand_2s, block=True)
