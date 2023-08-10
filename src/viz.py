# examples/Python/Basic/visualization.py

import numpy as np
import open3d as o3d
import copy


if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    mesh = o3d.io.read_triangle_mesh("STL/new_color.ply")
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

    pcd = pcd.uniform_down_sample(every_k_points=2)

    target = copy.deepcopy(pcd)
    target.remove_non_finite_points()

    source_points = np.asarray(target.points)
    # source_points = source_points - source_points.min(axis = 0)
    # source_points = source_points / source_points.max(axis = 0)
    # source_points = source_points * 2
    # source.points = o3d.utility.Vector3dVector(source_points)

    # source_points = np.asarray(source.points)
    # source_points = source_points - source_points.mean(axis = 0)

    target.points = o3d.utility.Vector3dVector(source_points)
    target.estimate_normals()
    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name="RPMNet", width=960, height=540, left=0, top=0)
    vis1.add_geometry(target)
    count = 0

    while count < 2000:
        # vis1.update_geometry(result)
        # vis1.update_geometry(result_gt)
        vis1.update_geometry(source)

        if not vis1.poll_events():
            break
        vis1.update_renderer()
        count += 1

    # else:
    #     print("The loaded point cloud does not have color information.")