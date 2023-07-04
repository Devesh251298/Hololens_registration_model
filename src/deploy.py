import os
import torch
import torch.utils.data
import numpy as np
from arguments import rpmnet_train_arguments
from common.misc import prepare_logger
from utils import get_model
from flask import Flask, jsonify, request


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

model = get_model(args, device, log_path)
model.eval()

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get the input data from the request
    # Perform any necessary preprocessing on the input data
    # Convert the input data to a PyTorch tensor
    input_tensor = data["data_batch"]
    types = data["data_types"]
    for key in input_tensor.keys():
        if types[key] == "<class 'torch.Tensor'>":
            input_tensor[key] = torch.tensor(input_tensor[key])
        elif types[key] == "<class 'numpy.ndarray'>":
            input_tensor[key] = np.array(input_tensor[key])
    # Run inference on the model
    with torch.no_grad():
        output_tensor, endpoints = model(input_tensor, args.num_reg_iter)
    # Convert the output tensor to a Python list
    output = output_tensor[0].cpu().detach().numpy()[0].tolist()
    print(output)

    # Create a response dictionary with the model's predictions
    response = {"output": output}
    # Return the response as JSON
    return jsonify(response)


if __name__ == "__main__":
    app.run()
