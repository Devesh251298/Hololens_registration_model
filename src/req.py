import requests
import json
import os
import torch
import torch.utils.data
import numpy as np
from arguments import rpmnet_train_arguments
from common.misc import prepare_logger
from data_loader.datasets import get_source, generate_data


# Set up arguments and logging
parser = rpmnet_train_arguments()
args = parser.parse_args()
logger, log_path = prepare_logger(args)
if args.gpu >= 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
else:
    device = torch.device("cpu")

source = get_source(args.object_file)
data_batch = generate_data(source, args)
data_types = {}

for key in data_batch.keys():
    # know the type of data_batch[key] and if it is a tensor or numpy array, convert it to a list 
    # store the types first
    data_types[key] = str(type(data_batch[key]))
    if isinstance(data_batch[key], torch.Tensor):
        data_batch[key] = data_batch[key].tolist()
    elif isinstance(data_batch[key], np.ndarray):
        data_batch[key] = data_batch[key].tolist()


data = {"data_batch": data_batch, "data_types": data_types}
json_data = json.dumps(data)

print("Data sent to the server:")

url = "http://localhost:5000/predict"  # Replace with your Flask route URL
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json_data)

transform = np.asarray(response.json()["output"])
transform = np.vstack((transform, np.array([0, 0, 0, 1])))
print(transform)


