import requests
import json
from data_loader.datasets import get_source
import os
import torch
import torch.utils.data
from arguments import rpmnet_train_arguments
from common.misc import prepare_logger


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

data = {"image": encoded_image}
json_data = json.dumps(data)

url = "http://localhost:5000/predict"  # Replace with your Flask route URL
headers = {"Content-Type": "application/json"}
response = requests.post(url, headers=headers, data=json_data)
