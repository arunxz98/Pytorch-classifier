import requests
import json
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image_path):
    """
    Preprocess the image for prediction
    """
    preprocess = transforms.Compose([
        transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension

    # Convert the tensor to float32 numpy array
    image = image.numpy().astype(np.float32)

    return image

def predict(image_path, server_url):
    """
    Send image to the MLflow model server for prediction
    """
    url = f"{server_url}/invocations"
    headers = {"Content-Type": "application/json"}

    # Preprocess the image
    image_data = preprocess_image(image_path)

    # Create payload
    payload = {
        "instances": image_data
    }

    # Send request
    response = requests.post(url, headers=headers, data=json.dumps(payload))

    if response.status_code == 200:
        predictions = response.json()
        print("Predictions:", predictions)
    else:
        print("Error:", response.text)


if __name__ == "__main__":
    image_path = "/home/arun/Documents/mlflow/Pytorch-classifier/images/val/EOSINOPHIL/_0_737.jpeg"  # Replace with the path to your image
    server_url = "http://127.0.0.1:5000"  # Replace with your server URL and port
    
    predict(image_path, server_url)
