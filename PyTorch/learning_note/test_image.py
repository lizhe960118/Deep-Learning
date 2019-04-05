import os
import pandas as pd
import numpy as np
from PIL import Image
import sys



df = pd.read_csv('./af2019-cv-training-20190312/list.csv')
print('Origin shape: ',df.shape)

df = df.dropna(axis=0)

num_of_df = df.shape[0]
print('Preprocessing: ', df.shape)

# image_ids = []

for index in range(num_of_df):
    image_path = str(df.ix[index,0])

    prefix_path = image_path[:2]

    image_path = './af2019-cv-training-20190312/' + prefix_path + '/' + image_path
    img_a = Image.open(image_path + '_a.jpg')
    img_b = Image.open(image_path + '_b.jpg')
    img_c = Image.open(image_path + '_c.jpg')

    im = Image.merge('RGB', (img_a, img_b, img_c)) 
    print(im.size)
    # r,g,b=img.split()
    break

img = Image.open("test.jpg")
print(img.shape)
r,g,b=img.split()       