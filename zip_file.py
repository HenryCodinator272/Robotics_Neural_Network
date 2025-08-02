import zipfile
import os
from PIL import Image
import numpy as np
#path = os.path.expanduser('~/Downloads/iCloud Photos/iCloud Photos')
'''
for file in os.listdir(path):
    with Image.open(os.path.join(path, file)) as image:
        if file.endswith('.JPG'):
            file = file.split('.')[0]
            image.save(f'images/rgb_images/{file}.png')
'''

'''
for file in os.listdir('images/rgb_images'):
    with Image.open(f'images/rgb_images/{file}') as img:
        img = img.resize((224, 224))
        img.save(f'images/rgb_images/{file}')
'''


for file in os.listdir('images/gt_images'):
    with Image.open(f'images/gt_images/{file}') as img:
        img = img.convert('RGB')
        array = np.array(img)
        array = np.transpose(array, (2, 0, 1))
        print(np.shape(array))
        output = np.zeros((np.shape(array)[1], np.shape(array)[2])).astype('uint8')
        print(np.shape(output))
        for y in range(len(array[0])):
            for x in range(len(array[0][y])):
                point = array[0][y][x]
                if 255 == array[0][y][x] == array[1][y][x] == array[2][y][x]:
                    output[y][x] = 2
                elif 0 == array[0][y][x] == array[1][y][x] == array[2][y][x]:
                    output[y][x] = 1

        Image.fromarray(output).save(f'images/processed_gt_images/{file}')

