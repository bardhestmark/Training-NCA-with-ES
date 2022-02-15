# @title Imports
import argparse
import pathlib

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import matplotlib.pyplot as plt
from torchsummary import summary
from model import CAModel


# @title Image manipulation
def load_image(path, size=40):
    """Load an image.
    Parameters
    ----------
    path : pathlib.Path
        Path to where the image is located. Note that the image needs to be
        RGBA.
    size : int
        The image will be resized to a square wit ha side length of `size`.
    Returns
    -------
    torch.Tensor
        4D float image of shape `(1, 4, size, size)`. The RGB channels
        are premultiplied by the alpha channel.
    """
    img = Image.open(path)
    img = img.resize((size, size), Image.ANTIALIAS)
    img = np.float32(img) / 255.0
    img[..., :3] *= img[..., 3:]

    return torch.from_numpy(img).permute(2, 0, 1)[None, ...]


def to_rgb(img_rgba):
    """Convert RGBA image to RGB image.
    Parameters
    ----------
    img_rgba : torch.Tensor
        4D tensor of shape `(1, 4, size, size)` where the RGB channels
        were already multiplied by the alpha.
    Returns
    -------
    img_rgb : torch.Tensor
        4D tensor of shape `(1, 3, size, size)`.
    """
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)


def make_seed(size, n_channels):
    """Create a starting tensor for training.
    The only active pixels are going to be in the middle.
    Parameters
    ----------
    size : int
        The height and the width of the tensor.
    n_channels : int
        Overall number of channels. Note that it needs to be higher than 4
        since the first 4 channels represent RGBA.
    Returns
    -------
    torch.Tensor
        4D float tensor of shape `(1, n_chanels, size, size)`.
    """
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x


# display img with shape of (c, M, N) of RGBA tensor
def display(img):
  """Display an image using pyplot
    ----------
    img : torch.Tensor
        3D tensor of shape (c, M, N) where M and N is the height and width
    -------
    """
  img_ = to_rgb(img)[0].numpy()
  c, m, n = img_.shape
  plt.figure(figsize=(15,5),facecolor='w')
  plt.axis("off")
  arr = np.ones((m,n,c))
  for i in range(c):
    arr[..., i] = img_[i]
  plt.imshow(arr)
  plt.show()

def drawfunc(img):
  """Create a numpy array from an img
    ----------
    img : torch.Tensor
        3D tensor of shape (3, M, N) where M and N is the height and width
    Returns
    -------
    numpy.ndarray
        3D float ndarray of shape `(c, M, N)`.
    """
  img_ = to_rgb(img)[0].numpy()
  c, m, n = img_.shape
  arr = np.ones((m,n,c))
  for i in range(c):
    arr[..., i] = img_[i]
  return arr


# TODO print parameters out
# Misc
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logdir = 'logs'
img = 'rabbit.png'
eps = np.finfo(float).eps
padding = 1
size = 40
batch_size = 4  # default 8
n_batches = 1000  # default 5000
pool_size = 8 # default 1024
n_channels = 16
eval_frequency = 50  # default 500
eval_iterations = 30  # default 300

# Hyperparameters
SIGMA = 1
LR = 0.001
MIN_ERROR_WEIGHT = 0.01
ERROR_WEIGHT = 1
DECAY_RATE = 0.95
POPULATION_SIZE = 1000
TOP_N = 200

# Logs
log_path = pathlib.Path(logdir)
log_path.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_path)

# Target image
target_img_ = load_image(img, size)
p = padding
target_img_ = nn.functional.pad(target_img_, (p, p, p, p), "constant", 0)
target_img = target_img_.to(device)
target_img = target_img.repeat(batch_size, 1, 1, 1)

writer.add_image("ground truth", to_rgb(target_img_)[0])

# Model and optimizer
model = CAModel(n_channels=n_channels, device=device)


# @title ES Module { run: "auto", form-width: "20%" }
# Fitness function

def batch_loss(x):
    """Run loss on batch.
        Parameters
        ----------
        x : torch.Tensor
            Shape `(batch_size, n_channels, size+padding, size+padding)`.
        Returns
        -------
        float
            shape (1) Inverse of loss (1/loss), shape(1,batch_size) Inverse of loss (1/loss) for each in batch
    """
    loss_batch = ((target_img - x[:, :4, ...]) ** 2).mean(dim=[1, 2, 3])  # Loss function
    loss = loss_batch.mean()
    return 1 / (loss + eps), 1 / loss_batch


def batch_fitness_func(x, solution):  # apply solution then calculate fitness
    """ Apply solution then calculate fitness.
      Parameters
      ----------
      x : torch.Tensor
          Shape `(batch_size, n_channels, size+padding, size+padding)`.
      solution : numpy.ndarray
          Shape (8320)
      Returns
      -------
      torch.Tensor
          Inverse of loss on new solution (1/loss)
      """
    # solution is a vector of paramters like mother_parametrs
    setMotherVector(solution)
    loss, _ = batch_loss(model(x))
    return loss + eps


def setMotherVector(solution):
    # set new params for model
    nn.utils.vector_to_parameters(solution, model.parameters())


# in ES, our population is a slightly altered version of the mother parameters, so we implement a jitter function
def jitter(mother_params, state_dict):
    """ Make a new parameter with specific noise.
      Parameters
      ----------
      mother_params : torch.Tensor
          Shape (8320)
      state_dict : torch.Tensor
          Shape (8320)
      Returns
      -------
      numpy.ndarray
          Shape(8320)
      """
    params_try = mother_params + ERROR_WEIGHT * SIGMA * state_dict
    return params_try


# now, we calculate the fitness of entire population
def batch_calculate_population_fitness(x, pop, mother_vector):
    """ Calculate population fitnesses.
      Parameters
      ----------
      x : torch.Tensor
          Shape `(batch_size, n_channels, size+padding, size+padding)`.
      pop : torch.Tensor
          Shape (POPULATION_SIZE, 8320)
      mother_vector : torch.Tensor
          Shape (8320)
      Returns
      -------
        (torch.Tensor, torch.Tensor)
          first of tuple with shape (POPULATION_SIZE, 1)
          and second of tuple with shape (POPULATION_SIZE, 8320)
      """
    fitness = torch.zeros(pop.shape[0], device=device)
    pop_weights = torch.zeros((pop.shape[0], pop.shape[1]), device=device)
    for i, params in enumerate(pop):
        p_try = jitter(mother_vector, params)
        pop_weights[i] = p_try
        fitness[i] = batch_fitness_func(x, p_try)
    return fitness, pop_weights


def es_step(x, mother_vector):
    """ Calculate one-step new parameters based on Evolutionary Strategies method
      Parameters
      ----------
      x : torch.Tensor
          Shape `(batch_size, n_channels, size+padding, size+padding)`.
      mother_vector : torch.Tensor
          Shape (8320)
      Returns
      -------
        torch.Tensor
          Shape (8320)
      """
    global ERROR_WEIGHT
    n_params = nn.utils.parameters_to_vector(model.parameters()).shape[0]
    # Create population in N(0, 1)
    pop = torch.rand(POPULATION_SIZE, n_params, device=device)
    # Get fitness for population and their population weights
    fitness, pop_weights = batch_calculate_population_fitness(x, pop, mother_vector)
    top_n_indices = torch.topk(fitness, TOP_N).indices
    F = fitness[top_n_indices]
    # Take top n fitness scores and weights
    F = (F - torch.mean(F)) / (torch.std(F) + eps)
    P = pop_weights[top_n_indices]
    # Update model parameters
    mother_vector += (LR / (TOP_N * SIGMA)) * torch.matmul(P.t(), F)
    # Decay error weight
    ERROR_WEIGHT = max(ERROR_WEIGHT * DECAY_RATE, MIN_ERROR_WEIGHT)
    if torch.nan in mother_vector:
        raise Exception('Values in mother_vector were torch.nan')
    return mother_vector


# @title {vertical-output:true}
# Model params initialization
n_params = nn.utils.parameters_to_vector(model.parameters()).shape[0]
print(f"Number of params: {n_params}")
mother_vector = nn.utils.parameters_to_vector(model.parameters())
print(mother_vector.shape)
summary(model, (16, 42, 42))
# Pool initialization
seed = make_seed(size, n_channels).to(device)
seed = nn.functional.pad(seed, (p, p, p, p), "constant", 0)
pool = seed.clone().repeat(pool_size, 1, 1, 1)

model = CAModel(n_channels=n_channels, device=device)
mother_vector = nn.utils.parameters_to_vector(model.parameters())

loss_log = []
loss_mean = []
model.train()
with torch.no_grad():
    for it in tqdm(range(n_batches)):
        # testing with pool size and batch size equals 1
        batch_ixs = np.random.choice(
            pool_size, batch_size, replace=False
        ).tolist()
        x = pool[batch_ixs]
        for i in range(np.random.randint(64, 96)):
            x = model(x)

        # take a step to optimize model parameters
        mother_vector = es_step(x, mother_vector)

        # calculate fitness of new mother vector and set mother vec
        setMotherVector(mother_vector)
        loss, loss_batch = batch_loss(x)
        loss_log.append(loss.detach().cpu())
        # Make reward graph
        if it % 50 == 0 and it != 0:
            # output.clear()
            rm = np.mean(loss_log[50:])
            loss_mean.append((it, rm))
            plt.figure(figsize=(15, 5), dpi=80)
            plt.ylabel('Reward (log)')
            plt.xlabel('Training steps')
            plt.plot(loss_log, '-', alpha=0.7, linewidth=4)
            plt.plot(*zip(*loss_mean), 'r-')
            plt.yscale('log')
            plt.show()
            print(f"Iteration: {it}, Reward:{loss}, Last 50 reward mean:{rm}")
            display(x[:, :4, ...].detach().cpu())

        # Pool stuff
        argmin_batch = loss_batch.argmin().item()  # find the batch with lowest reward
        argmin_pool = batch_ixs[argmin_batch]
        remaining_batch = [i for i in range(batch_size) if i != argmin_batch]  # remove arg min batch from batches
        remaining_pool = [i for i in batch_ixs if i != argmin_pool]  # remove arg min pool from pools
        pool[argmin_pool] = seed.clone()  # remove in pool the one with lowest score
        pool[remaining_pool] = x[remaining_batch].detach()  # set pool to remaining ones

        """
        # Evaluation video for tensorboard
        if it % eval_frequency == 0:
            x_eval = seed.clone()  # (1, n_channels, size, size)
  
            eval_video = torch.empty(1, eval_iterations, 3, *x_eval.shape[2:])
  
            for it_eval in range(eval_iterations):
                x_eval = model(x_eval)
                x_eval_out = to_rgb(x_eval[:, :4].detach().cpu())
                eval_video[0, it_eval] = x_eval_out
  
            writer.add_video("eval", eval_video, it, fps=60)
        """

# @title Animation { vertical-output: true, form-width: "40%" }
# comment out to not skip

import matplotlib.animation as animation
from matplotlib import rc

rc('animation', html='jshtml')
model.training = False
fig, ax = plt.subplots()

ims = []
x = pool[0:1]
for i in range(100):
    if i == 0:
        ims.append([ax.imshow(drawfunc(x[:, :4, ...].detach().cpu()))])
    x = model(x)
    im = drawfunc(x[:, :4, ...].detach().cpu())
    im = ax.imshow(im)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
f = r"./animation.mp4"
writervideo = animation.FFMpegWriter(fps=60)
ani.save(f, writer=writervideo)
