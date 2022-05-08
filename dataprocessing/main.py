import numpy as np
from utils import convert_tb_data
import tensorboard as tb
import matplotlib.pyplot as plt
import os
from colorama import Fore
plt.style.use('seaborn-whitegrid')

dirs = ['..\interactive_CA\logs', '..\\final_models']
logdir = dirs[0]
eventdirs = [d for d in os.listdir(logdir)]

for dir in eventdirs:
    path = f"{logdir}/{dir}"

    try:
        df = convert_tb_data(path)
    except ValueError as ve:
        print(Fore.RED+f'A directory does not include a tensorboard file:{ve}')
        Fore.WHITE

    fig = plt.figure()
    ax = plt.axes()
    ax.plot(df['step'], df['value'])
    plt.show()
