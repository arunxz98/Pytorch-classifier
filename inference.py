import torch
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import InterpolationMode
import time
import os
import glob
from PIL import Image
import csv
from tqdm import tqdm
import json

from base_model import MobNetv2_custom_classes



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_transform = transforms.Compose([
        transforms.Resize(224,transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


label_maps = json.load(open("label_maps.json"))
configs = json.load(open("config.json"))


# the below two lines of code are used if we save model state instead of saving along with graph
# model = MobNetv2_custom_classes()
# model.load_state_dict(copy.deepcopy(torch.load(model_path,map_location=device)))

# loading the model
model = torch.load(configs['trained_model_path'],map_location=device)
model.eval()

# output variables to write to output file
output_header = ["image_name","class_name","gt_class_index","pred[0]","pred[1]","pred[2]","pred[3]","pred_class","latency"]

with torch.no_grad(): #no need for grads in evaluation phase
    with open(configs["pred_output"],"w") as f:
        writer = csv.DictWriter(f,fieldnames=output_header)
        writer.writeheader()

    # looping through images for prediction
        folders = os.listdir(configs['test_dir_path'])
        for folder in folders:
            """folder is also considered name.
            So, when use of classname is required folder(str) variable is used"""
            images_list = glob.glob(os.path.join(
                os.path.abspath(configs['test_dir_path']),
                folder,
                "*jpeg"
            ))
            print(f'evaluating{folder} class')
            for each_image in tqdm(images_list):
                frame = Image.open(each_image)
                frame = data_transform(frame)
                start = time.time()
                pred = model(frame.unsqueeze(0)) # image dims will be [1,3,224,224] with unsqueeze
                latency = time.time()-start
                writer.writerow({
                    "image_name": os.path.basename(each_image),
                    "class_name": str(folder),
                    "gt_class_index":label_maps[str(folder)],
                    "pred[0]":pred[0][0].item(),
                    "pred[1]":pred[0][1].item(),
                    "pred[2]":pred[0][2].item(),
                    "pred[3]":pred[0][3].item(),
                    "pred_class":torch.argmax(pred).item(),
                    "latency":latency
                })

