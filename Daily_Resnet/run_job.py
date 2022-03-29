from turtle import forward
from nbox import Operator
from nbox.nbxlib.ops import ShellCommand
from setuptools import Command
import datetime

def hr():
    print("-" * 80)

class GetImages(Operator):
    def __init__(self):
        super().__init__()
        self.n = 10
        self.folder = "new_test"
        command = f"python get_images.py --n {self.n} --folder {self.folder}"
        self.shell = ShellCommand(command)
    def forward(self):
        self.shell()

class TrainResnet(Operator):
    def __init__(self) -> None:
        super().__init__()
        self.num_epochs = 20
        self.img_folder = "new_test"
        self.save_folder = "daily_resnet"
        command = f"python train.py --num_epochs {self.num_epochs} --img_folder {self.img_folder} --save_folder {self.save_folder}"
        self.shell = ShellCommand(command)
    def forward(self):
        self.shell()
        

class DeployResnet(Operator):
    def __init__(self) -> None:
        super().__init__()
        time_now = datetime.datetime.now()
        self.model_file_name = f"{time_now.date()}-{time_now.hour}.pth"
        command = f"python deploy_resnet.py --model_file_name {self.model_file_name}"
        self.shell = ShellCommand(command)
    def forward(self):
        self.shell()

class DailyResnet(Operator):
    def __init__(self) -> None:
        super().__init__()
        self.image_getter = GetImages()
        self.resnet_trainer = TrainResnet()
        self.deployer = DeployResnet()

    def forward(self):
        hr()
        print("Getting Images")
        hr()
        self.image_getter()
        print("Training Resnet")
        hr()
        self.resnet_trainer()
        print("Deploying Resnet")
        hr()
        self.deployer()

# if __name__ == "__main__":
#     daily_resnet = DailyResnet()
#     daily_resnet()
#     print("Job Successful")