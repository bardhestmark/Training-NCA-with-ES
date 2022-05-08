import numpy as np
from utils import convert_tb_data
import tensorboard as tb
import matplotlib.pyplot as plt
import os
import time
from colorama import Fore
plt.style.use('seaborn-whitegrid')

dirs = ['data\\growing_results\\non_samples', 'data\\growing_results\\samples']
size = ['9x9', '15x15']
type_ = ['carrot', 'rabbit']
colors = ['r','b','g','m','c']
labels = ['ES', 'Adam']

s = size[0]
t = type_[1]

logdir = dirs[0]+os.sep+s+os.sep+t
eventdirs = [d for d in os.listdir(logdir)]
fig = plt.figure()
ax = plt.axes()
ax.set_yscale('log')
ax.set_title(f'{s} {t}')
ax.set_xlabel('Update steps')
ax.set_ylabel(r'log$_1$$_0$(loss)')
for dir in eventdirs:
    path = f"{logdir}/{dir}"

    try:
        df = convert_tb_data(path)
    except ValueError as ve:
        print(Fore.RED+f'A directory does not include a tensorboard file:{ve}')
        Fore.WHITE

    ax.plot(df['step'], df['value'],color=colors.pop(), label=labels.pop())

#plt.show()
plt.legend()
plt.savefig(f'graphs/{s}-{t}_{time.time()}.png', dpi=400)
