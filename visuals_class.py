import matplotlib.pyplot as plt
import os
import numpy as np

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

