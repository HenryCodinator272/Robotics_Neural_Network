import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

def plot_loss(train, evaluation, current_epoch):

    x = np.arange(0, current_epoch + 1)
    y1 = train
    y2 = evaluation

    if max(train) > max(evaluation):
        max_y = max(train)
    else:
        max_y = max(evaluation)


    plt.grid(True)

    plt.plot(x, y1, color = 'r')
    plt.plot(x, y2, color = 'b')
    plt.xlim(0, len(train))
    plt.ylim(0, 1.1 * max_y)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'eval'])

    os.makedirs('graph_results', exist_ok = True)
    plt.savefig('graph_results/loss.png')
    plt.close()

def stitch_images(file_number, mask_image):
    for file in os.listdir('saved_images'):
        os.remove(f'saved_images/{file}')
    with Image.open(f'images/rgb_images/IMG_{file_number}') as img:
        img = img.convert('L')
        visual = np.array(img)
        combined_array = np.hstack([mask_image, visual])
        Image.fromarray(combined_array).save(f'saved_images/image_{file_number}')


