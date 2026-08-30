import kagglehub
from pathlib import Path
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def random_colour():
    colours = ( 
        random.randint(0,255),
        random.randint(0,255),
        random.randint(0,255)
    )

    return colours

def draw_heart(path):
    heart_colour = random_colour()
    img = Image.new("RGB", (224,224), (255,255,255))
    img_draw = ImageDraw.Draw(img)

    scale = random.uniform(0.7, 1.2)

    offset_x = random.randint(-20,20)
    offset_y = random.randint(-20,20)

    points=[]


    # Heart using parametric equation
    for t in np.linspace(0, 2*np.pi, 300):
        size=224
        x = 16*np.sin(t)**3

        y = (
            13*np.cos(t)
            -5*np.cos(2*t)
            -2*np.cos(3*t)
            -np.cos(4*t)
        )


        px = (
            size/2
            + x*5*scale
            + offset_x
        )

        py = (
            size/2
            - y*5*scale
            + offset_y
        )

        points.append(
            (px,py)
        )

    img_draw.polygon(
        points,
        fill=heart_colour
    )

    img.save(path)


kaggle_dataset = "khalidboussaroual/2d-geometric-shapes-17-shapes"
dataset_name = kaggle_dataset.split("/")[-1]    # To keep the dataset name 
save_dataset = Path.cwd() / "data" / dataset_name
save_dataset.mkdir(parents=True, exist_ok=True)

path = kagglehub.dataset_download(
    kaggle_dataset,
    output_dir = str(save_dataset)
)

print("Creating hearts")
heart_data = save_dataset / "2D_Geometric_Shapes_Dataset" / "heart"
os.makedirs(heart_data, exist_ok=True)
for i in tqdm(range(50000)):
    file_name = f"{i}.png"

    file_path = os.path.join(heart_data, file_name)
    draw_heart(file_path)

print("File are here: ", Path(path).resolve())    # To print the full path
