import os
import random
import numpy as np
from PIL import Image, ImageDraw

heart_data = "heart"

os.makedirs(heart_data, exist_ok=True)

def draw_heart(path):
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
        fill="black"
    )

    img.save(path)


for i in range(2000):
    file_name = f"{i}.png"

    file_path = os.path.join(heart_data, file_name)
    draw_heart(file_path)