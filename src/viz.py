# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d
import copy


if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    mesh = o3d.io.read_triangle_mesh("STL/skull.stl")
    vertices = np.asarray(mesh.vertices)

    scale_factor = 0.001
    scaled_vertices = vertices * scale_factor

    mesh.vertices = o3d.utility.Vector3dVector(scaled_vertices)
    mesh.compute_vertex_normals()

    pcd = o3d.geometry.PointCloud()
    pcd.points = mesh.vertices
    pcd.colors = mesh.vertex_colors
    pcd.normals = mesh.vertex_normals

    pcd = pcd.uniform_down_sample(every_k_points=2)

    target = copy.deepcopy(pcd)
    target.remove_non_finite_points()

    source_points = np.asarray(target.points)
    target.points = o3d.utility.Vector3dVector(source_points)
    target.estimate_normals()
    print(np.asarray(target.points))
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="RPMNet", width=960, height=540, left=0, top=0)
    vis1.add_geometry(target)
    count = 0

    while count < 2000:
        vis1.update_geometry(target)

        if not vis1.poll_events():
            break
        vis1.update_renderer()
        count += 1
