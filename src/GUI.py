import PySimpleGUI as sg
import subprocess
import pyvista as pv
import os

# ... (rest of the code)

# Define the layout of the GUI with default values
# Define the layout of the GUI with default values
layout = [
    [sg.Radio("Evaluate", "mode", default=True, key="mode_eval"), sg.Radio("Train", "mode", key="mode_train")],
    [sg.Text("Input STL File"), sg.InputText(key="input_file", default_text="STL/Segmentation_skin.stl"), sg.FileBrowse()],
    [sg.Text("Input Target File"), sg.InputText(key="target_file", default_text="STL/hololens_mesh.ply"), sg.FileBrowse()],
    [sg.Text("Number of Points"), sg.InputText(key="num_points", default_text="12000")],
    [sg.Checkbox("Visualize", default=True, key="visualize")],
    [sg.Text("Model Path"), sg.InputText(key="model_path", default_text="models/model-new_best.pth"), sg.FileBrowse()],
    [sg.Text("Number of Planes"), sg.InputText(key="num_planes", default_text="0")],
    [sg.Checkbox("Save Mesh", default=True, key="save_mesh")],
    [sg.Checkbox("Capture RGB", default=True, key="capture_rgb")],
    [sg.Checkbox("Simulated", default=True, key="simulated")],
    [sg.Button("Show STL"), sg.Button("Show Target"), sg.Button("Run"), sg.Button("Exit")]
    ]


def display_3d_model(file_path):
    if os.path.exists(file_path):
        mesh = pv.read(file_path)
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color='w')
        plotter.show()
    else:
        sg.popup("File not found.", title="Error")

def show_3d_model(file_path):
    cmd = ["python", "3d_viewer.py", file_path]  # Replace with the correct command
    subprocess.run(cmd)

# Create the window
window = sg.Window("Function Executor", layout)
prev_event = None

# Event loop
while True:
    event, values = window.read()
    print(event, prev_event)
    if (event == sg.WIN_CLOSED and prev_event=="Run") or event == "Exit":
        break
    elif event == "Run":
        prev_event = event
        if values["mode_train"]:
            cmd = [
                "python", "train.py",
                "--object_file", values["input_file"],
                "--target_file", values["target_file"],
                "--num_point", values["num_points"],
                "--visualize", str(values["visualize"]),
                "--resume", values["model_path"],
                "--num_planes", values["num_planes"],
                "--save_mesh", str(values["save_mesh"]),
                "--capture_rgb", str(values["capture_rgb"]),
                "--simulated", str(values["simulated"]),
            ]
        else:  # mode_eval
            cmd = [
                "python", "eval.py",
                "--object_file", values["input_file"],
                "--target_file", values["target_file"],
                "--num_point", values["num_points"],
                "--visualize", str(values["visualize"]),
                "--resume", values["model_path"],
                "--num_planes", values["num_planes"],
                "--save_mesh", str(values["save_mesh"]),
                "--capture_rgb", str(values["capture_rgb"]),
                "--simulated", str(values["simulated"]),
            ]
        
        try:
            subprocess.run(cmd, check=True)
            sg.popup("Execution completed successfully!", title="Success")
        except subprocess.CalledProcessError:
            sg.popup("An error occurred during execution.", title="Error")
    elif event == "Show STL":
        stl_path = values["input_file"]
        show_3d_model(stl_path)
    elif event == "Show Target":
        target_path = values["target_file"]
        show_3d_model(target_path)


# Close the window when the loop is exited
window.close()
