import nbox
from nbox import Operator
import os
import json
import torch
from torchvision import transforms
from PIL import Image

def pre_fn(x):
    return {"x": x}

def hr():
    print("\n", "="*100, "\n")

class ImageInfer(Operator):
    def __init__(self, images, model_path=None):
        super().__init__()
        if model_path in [None, "", "None"]:
            self.model = nbox.load("torchvision/resnet18", pretrained=True)
        else:
            m = torch.load(model_path)
            self.model = nbox.Model(m)
        self.images = images
        labels = json.load(open("imagenet_class_index.json"))
        self.idx2label = [labels[str(k)][1] for k in range(len(labels))]

    def forward(self):
        infer = []
        self.model.model.eval()
        for name, img in self.images:
            out = self.model(pre_fn(img))
            _, idx = torch.max(out, 1)
            percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
            infer.append([name, self.idx2label[idx[0]], percentage[idx[0]].item()])
        return infer

        
class Images(Operator):
    def __init__(self, parent, img_names, model_path=None):
        super().__init__()
        self.images = []
        for name in img_names:
            loader = transforms.Compose([transforms.Scale((224, 224)), transforms.ToTensor(), ])
            img = Image.open(f"./{parent}/{name}")
            img = img.convert("RGB")
            img = loader(img).float()
            img = img.unsqueeze(0)
            self.images.append([name, img])
        
        self.Inferencer = ImageInfer(self.images, model_path=model_path)
    
    def forward(self):
        inference = self.Inferencer()
        for item in inference:
            hr()
            print(f"File Name: {item[0]} -- Prediction: {item[1]} -- Confidence: {item[2]}%")

