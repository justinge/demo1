# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Perform test request
"""

import pprint

import requests

DETECTION_URL = "http://192.168.47.193:5000/v1/img_url"
IMAGE = "../../data/images/zidane.jpg"

# Read image
with open(IMAGE, "rb") as f:
    image_data = f.read()
print(image_data)
response = requests.post(DETECTION_URL, json={"img_url": "http://192.168.47.193:5000/image"}).json()

pprint.pprint(response)
