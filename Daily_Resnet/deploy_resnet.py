#Imports
from logging import raiseExceptions
from tkinter import Variable
import nbox
import torch
from PIL import Image
from argparse import ArgumentParser
import datetime
from torchvision import transforms

def pre_fn(x):
    return {"x": x}
if __name__ == "__main__":
    args = ArgumentParser("Deploy Resnet18")
    args.add_argument("--model_file_name", type=str)
    args = args.parse_args()
    time_now = datetime.datetime.now()
    model_file_name = args.model_file_name #f"{time_now.date()}-{time_now.hour}.pth"
    model_path = f"daily_resnet/{model_file_name}"
    img_add = f"random_images/0.jpg"

    #Load the model using nbox
    model = torch.load(model_path)
    m = nbox.Model(
        model,
        pre = pre_fn

    )
    loader = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor()])
    img = Image.open(img_add)
    img = loader(img).float()
    img = img.unsqueeze(0)
    #Check if the model is able to infer
    try:       
        out = m(img)
        #print(torch.argmax(out[0]))
    except:
        raise ValueError("The model is unable to infer the given image")
    
    #Deploy
    try:
        #Change this line to deploy the model
        #m.deploy(input_object=img)
        print("Model is not Deployed yet, modify the code in deploy_resnet.py to deploy the model")
    except:
        raise ValueError("The model is unable to deploy")
