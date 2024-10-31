# test/test_request.py

import base64
import requests

# Convert the image to base64 format
with open("path_to_test_image.jpg", "rb") as img_file:
    base64_string = base64.b64encode(img_file.read()).decode('utf-8')

# Send the base64 image to the server for prediction
response = requests.post('http://127.0.0.1:5000/predict', json={'image': base64_string})
print(response.json())
