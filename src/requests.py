import requests
import base64
import json

with open('path/to/your/image.jpg', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

data = {'image': encoded_image}
json_data = json.dumps(data)

url = 'http://localhost:5000/predict'  # Replace with your Flask route URL
headers = {'Content-Type': 'application/json'}
response = requests.post(url, headers=headers, data=json_data)
