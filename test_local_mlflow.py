import mlflow.pytorch
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image_path):
    """
    Preprocess the image for prediction
    """
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def load_model(model_uri):
    """
    Load the model from MLflow
    """
    model = mlflow.pytorch.load_model(model_uri)
    return model

def predict(image_path, model_uri):
    """
    Load the model and make predictions on the input image
    """
    # Preprocess the image
    image_tensor = preprocess_image(image_path)

    # Load the model
    model = load_model(model_uri)

    # Set the model to evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        return preds

if __name__ == "__main__":
    image_path = "images/train/LYMPHOCYTE/_0_331.jpeg"  # Replace with the path to your image
    model_uri = "mlruns/160800716032786696/db09555f3d0544fe904d7fb6d66288fb/artifacts/best_model"  # Update with your model URI
    
    predictions = predict(image_path, model_uri)
    print("Predicted class:", predictions.item())
