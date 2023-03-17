import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image



classes = ['Ramnit', 'Lollipop', 'Kelihos_ver3', 'Vundo', 'Simda', 'Tracur', 'Kelihos_ver1', 'Obfuscator.ACY', 'Gatak']
labels = [i for i in range(1, 10)]
images = []
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Ramnit.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Lollipop.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Kelihos_ver3.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Vundo.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Simda.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Tracur.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Kelihos_ver1.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Obfuscator.ACY.png'))
images.append(plt.imread('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/Gatak.png'))


# Load the data in trainLabels.csv
data = pd.read_csv('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/trainLabels.csv')

#print(data['Class'])

imgs = data['Class'].apply(lambda x: images[x-1])


#img_width = 64
#img_height = 64
#rows = 1
#cols = 10868
#
#sprite_image = Image.new('RGB', (img_width * cols, img_height * rows))
#
#col = 1
#for img in imgs:
#    # convering img to 3 channels (RGB) instead of 4 (RGBA)
#    img = img[:,:,:3]
#    img = Image.fromarray((img * 255).astype(np.uint8))
#    # save the image
#    #img.save('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/img_' + str(col) + '.png')
#    sprite_image.paste(img, (img_width * (col-1), 0))
#    col += 1
#
#sprite_image.save('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/sprite_image.png')

num_images = len(imgs)
print(num_images)
num_cols = int(np.ceil(num_images / 120))

img_width = 64
img_height = 64
rows = 120
cols = num_cols

#sprite_image = Image.new('RGB', (img_width * cols, img_height * rows))
#
#row = 0
#col = 0
#for img in imgs:
#    # convering img to 3 channels (RGB) instead of 4 (RGBA)
#    img = img[:,:,:3]
#    img = Image.fromarray((img * 255).astype(np.uint8))
#    # save the image
#    img.save('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/images/img_' + str(row*cols+col+1) + '.png')
#    sprite_image.paste(img, (col * img_width, row * img_height))
#    col += 1
#    if col == cols:
#        col = 0
#        row += 1
#
#sprite_image.save('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/sprite_image.png')

sprite_image = Image.new('RGBA', (img_width * cols, img_height * rows), (0, 0, 0, 0))

row = 0
col = 0
for img in imgs:
    # convering img to 4 channels (RGBA)
    img = img[:,:,:4]
    img = Image.fromarray((img * 255).astype(np.uint8))
    # paste the image onto the sprite image with the alpha channel intact
    sprite_image.paste(img, (col * img_width, row * img_height))
    col += 1
    if col == cols:
        row += 1
        col = 0

sprite_image.save('/Users/acburihan/Documents/IMTA/MalVis/TenMalVis/sprite_image.png')
