import requests
import json
import os
import torch
import torch.utils.data
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

data = {"data_batch": data_batch}
json_data = json.dumps(data)

url = "http://localhost:5000/predict"  # Replace with your Flask route URL
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json_data)

print(response.json())
