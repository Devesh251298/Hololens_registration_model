import os
from stl import mesh
import numpy as np

# Define the folder containing the STL files
folder_path = "stl_files1"

# Define the output STL file
output_stl_file = "STL/brain.stl"

# Initialize an empty list to store mesh objects
mesh_list = []

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".stl"):
        # Load each STL file as a mesh
        stl_mesh = mesh.Mesh.from_file(os.path.join(folder_path, filename))
        
        # Append the mesh to the list
        mesh_list.append(stl_mesh)

# Combine all the loaded meshes into a single mesh
combined_mesh = mesh.Mesh(np.concatenate([mesh.data for mesh in mesh_list]))

# Save the combined mesh to an output STL file
combined_mesh.save(output_stl_file)
