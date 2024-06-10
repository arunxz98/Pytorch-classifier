import time
import torch
from torchvision import transforms
from PIL import Image
import json
import io
from flask import Flask, request, jsonify

# Load your custom model
from base_model import MobNetv2_custom_classes

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image transformations
data_transform = transforms.Compose([
    transforms.Resize(224, transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load label maps and configurations
label_maps = json.load(open("label_maps.json"))
configs = json.load(open("config.json"))

# Load the model
model = torch.load(configs['trained_model_path'], map_location=device)
model.eval()

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        image = Image.open(io.BytesIO(file.read()))
        image = data_transform(image).unsqueeze(0)  # Transform and add batch dimension

        with torch.no_grad():
            start_time = time.time()
            pred = model(image)
            latency = time.time() - start_time

            pred_class = torch.argmax(pred).item()
            class_name = list(label_maps.keys())[list(label_maps.values()).index(pred_class)]

            return jsonify({
                "class_name": class_name,
                "pred_class": pred_class,
                "latency": latency,
                "predictions": {
                    "pred[0]": pred[0][0].item(),
                    "pred[1]": pred[0][1].item(),
                    "pred[2]": pred[0][2].item(),
                    "pred[3]": pred[0][3].item()
                }
            })

# Main block to run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
