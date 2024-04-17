import matplotlib.pyplot as plt
import numpy as np

def plot_one_loss(epoch, loss, legend, mark, city):
    plt.plot(epoch, loss, 'r')
    plt.legend([f'{legend}'])
    plt.savefig(f'./figures/{city}_training_process_{mark}.jpg')
    plt.clf()
    plt.close()
    
def plot_two_loss(epoch, loss1, loss2, legend1, legend2, mark, city):
    plt.plot(epoch, loss1, 'b')
    plt.plot(epoch, loss2, 'r')
    plt.legend([f'{legend1}', f'{legend2}'])
    plt.savefig(f'./figures/{city}_training_process_{mark}.jpg')
    plt.clf()
    plt.close()