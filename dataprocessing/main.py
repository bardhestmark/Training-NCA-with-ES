import numpy as np
from utils import convert_tb_data
import tensorboard as tb
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')

logdir = 'interactive_CA\logs'
eventpath = 'ADAM-RABBIT FACE_07-05-2022_14-34-42'
path = f"{logdir}/{eventpath}"
df = convert_tb_data(path)

fig = plt.figure()
ax = plt.axes()
ax.plot(df['step'], df['value'])
plt.show()
