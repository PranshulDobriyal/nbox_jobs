#Script to download specified number of random images and save them in specified folder

#imports
import requests
from argparse import ArgumentParser
import random
import os
    
if __name__ == "__main__":
    args = ArgumentParser(description="Script to get some images")
    args.add_argument("--n", type=int, help="Number of Images")
    args.add_argument("--base_address", type=str, help="Base Address", default="")
    args.add_argument("--folder", type=str, help="Name of the folder where Images are to be stored")
    args = args.parse_args()
    
    if args.base_address == "":
        folder_path = args.folder
    else:
        folder_path = f"{args.base_address}/{args.folder}"
    #Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)
    for i in range(args.n):
        #Generate the file name
        filepath = os.path.join(folder_path, f"{str(i)}.jpg")
        #Generate keys to get different images on every iteration
        key = random.randint(100, 10000)
        url = f"https://picsum.photos/seed/{str(key)}/300"
        response = requests.get(url, filepath)
        with open(filepath, "wb") as f:
            f.write(response.content)
