#import
import nbox
import random
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms 
import os 
from os.path import isfile
from PIL import Image
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
import numpy as np
import datetime
import wandb


class ImageDataset(Dataset):
    #Custom Dataset class
    def __init__(self, img_dir, tfms=None, train=True):
        self.img_dir = img_dir
        self.tfms = tfms
        self.img_names = [f for f in os.listdir(img_dir) if isfile(os.path.join(img_dir, f))]
        if not train:
            self.img_names = [self.img_names[-1]]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)
        label = torch.tensor(random.randint(1, 100)).unsqueeze(0)
        if self.tfms:
            image = self.tfms(image).unsqueeze(0)
        item = {"img":image, "label":label}
        return item


class TrainerConfig:
    def __init__(self, **kwargs):
        self.train_steps = int(1e7)
        self.gas = 1
        self.test_every = 100
        self.max_test_steps = 300
        for k, v in kwargs.items():
            setattr(self, k, v)


class Trainer:
    def __init__(
        self,
        model,
        train_conf,
        train_dl,
        test_dl,
        optim,
        lr_scheduler=None,
    ):
        self.model = model.model
        self.c = train_conf
        self.train_dl = train_dl
        self.test_dl = test_dl
        self.optim = optim
        self.lr_scheduler = lr_scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.device = "cpu"
        if torch.cuda.is_available():
            print("Model is now CUDA!")
            self.device = torch.cuda.current_device()
            if torch.cuda.device_count() > 1:
                print("Model is now DataParallel!")
                self.model = torch.nn.DataParallel(self.model)
            self.model = self.model.to(self.device)

        self.device = next(iter(self.model.parameters())).device

    def __call__(self, x, train=False, optim=None, lr_scheduler=None, loss_scale=1.0):
        m = self.model
        logger = {}
        x = {k: v.to(self.device) for k, v in x.items()}
        running_acc = 0
        # different conditionals for inference and training
        if not train:
            # inference is priority
            m.eval()
            with torch.no_grad():
                out = m(x["img"])
                loss = self.criterion(out, x["label"])
            return loss
        else:
            # just because this is training does not mean optimizer needs to be
            # provided. optimizer and lr_scheduler can be called whenever we want
            if optim != None:
                assert hasattr(optim, "step"), "Provide correct optimizer"
                if lr_scheduler != None:
                    assert hasattr(lr_scheduler, "step"), "Provide correct LR Scheduler"
            m.train()

            # forward pass
            out = m(x["img"])
            loss = CrossEntropyLoss()(out,x["label"])
            _,preds = torch.max(out,1)
            running_acc += torch.sum(preds==x["label"].data)
            loss.backward()

            if optim != None:
                optim.step()
                m.zero_grad()

            # if lr_scheduler is not None:
            #     wandb.log({"loss":loss.item(), "lr": lr_scheduler.get_lr(), "acc": self.running_acc.double() / self.[phase]})
            # else:
            #     wandb.log({"loss":loss.item()})
            logger["train_loss"] = loss.item() * x["img"].size(0)
            logger["train_acc"] = running_acc
            return logger

    def train(self):
        c = self.c
        train_dl = self.train_dl
        test_dl = self.test_dl
        optim = self.optim
        lr_scheduler = self.lr_scheduler
        iter_train_dl = iter(train_dl)
        all_losses = {"train": [], "test": []}
        all_acc_train = []
        pbar = trange(c.train_steps)
        for i in pbar:
            if i:
                desc_str = f"{all_losses['train'][-1]:.3f}"
                pbar.set_description(desc_str)

            try:
                x = next(iter_train_dl)
            except StopIteration:
                train_dl = self.train_dl
                iter_train_dl = iter(train_dl)
                x = next(iter_train_dl)

            logger = self(x, train=True, optim=optim, lr_scheduler=lr_scheduler, loss_scale=c.gas)
            all_losses["train"].append(logger["train_loss"])
            all_acc_train.append(logger["train_acc"])
            if lr_scheduler is not None:
                wandb.log({"loss":all_losses["train"][-1], "lr": lr_scheduler.get_lr(), "acc": all_acc_train[-1]})
            else:
                wandb.log({"loss":all_losses["train"][-1], "acc": all_acc_train[-1]})


            if (i + 1) % c.test_every == 0:
                test_losses = []
                iter_test_dl = iter(test_dl)
                for _, x in zip(range(c.max_test_steps), iter_test_dl):
                    # test_dl can be very big so we only test a few steps
                    _tl = self(x, False)
                    test_losses.append(_tl.item())
                test_loss = np.mean(test_losses)
                logger["test_loss"] = test_loss
                wandb.log({"test_loss": logger["test_loss"]})


if __name__ == "__main__":
    args = ArgumentParser(description="Script to get some images")
    #args.add_argument("--base_address", type=str, help="Address of parent directory ", default="")
    args.add_argument("--num_epochs", type=int, help="Number of epochs you want to train the models for")
    args.add_argument("--img_folder", type=str, help="Name of the directory where Images are stored")
    args.add_argument("--save_folder", type=str, help="Name of the directory where model is to be saved")
    args.add_argument("--model", type=str, default="torchvision/resnet18", help="Model you want to train")
    args = args.parse_args()

    image_size = (224, 224)
    transform = transforms.Compose([
                            transforms.Resize(image_size),
                            transforms.ColorJitter(hue=.05, saturation=.05),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomRotation(20, resample=Image.BILINEAR),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])

    model = nbox.load(args.model, pretrained = True)
    #base_address = f"{args.base_address}"
    optimizer = optim.Adam(model.model.parameters(), lr=1e-3)
    # train_data = ImageDataset(f"{base_address}/{args.img_folder}", tfms=transform, train=True)
    # test_data = ImageDataset(f"{base_address}/{args.img_folder}", tfms=transform, train=False)
    train_data = ImageDataset(f"{args.img_folder}", tfms=transform, train=True)
    test_data = ImageDataset(f"{args.img_folder}", tfms=transform, train=False)
    tc = TrainerConfig(
        train_steps=args.num_epochs,
        test_every=1,
    )
    t = Trainer(
        model=model,
        train_conf=tc,
        train_dl=train_data,
        test_dl=test_data,
        optim=optimizer,
    )
    #os.environ['WANDB_MODE'] = 'offline'
    wandb.init(
        project = "Daily ResNet18",
        tags = ["test-airflow"],
        config=tc
    )
    t.train()
    os.makedirs(f"{args.save_folder}", exist_ok=True)
    time_now = datetime.datetime.now()
    model_file_name = f"{time_now.date()}-{time_now.hour}.pth"
    torch.save(model.model, os.path.join(f"{args.save_folder}", model_file_name))
