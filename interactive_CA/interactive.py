
import time

import numpy as np
import pygame
import torch
import torch.nn.functional as F
from PIL import Image
from torch import tensor as tt
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from model import CAModel, CellularAutomataModel
from utils import load_emoji, to_rgb, adv_attack


class Interactive:
    def __init__(self, args):
        self.n_channels = args.n_channels
        self.hidden_size = args.hidden_size
        self.fire_rate = args.fire_rate
        self.size = args.size + 2 * args.padding
        self.logdir = args.logdir
        self.es = args.es
        self.eps = args.eps

        if self.es: self.target_img = load_emoji(args.img, self.size)
        else: self.target_img = torch.from_numpy(load_emoji(args.img, self.size)).permute(2, 0, 1)[None, ...]
        self.writer = SummaryWriter(self.logdir)
        p = args.padding

        if self.es:
            self.pad_target = F.pad(tt(self.target_img), (0, 0, p, p, p, p))
            self.net = CellularAutomataModel(n_channels=self.n_channels, fire_rate=self.fire_rate, hidden_channels=self.hidden_size)
            h, w = self.pad_target.shape[:2]
            self.seed = np.zeros([h, w, self.n_channels], np.float64)
            self.seed[h // 2, w // 2, 3:] = 1.0
        else:
            self.net = CAModel(n_channels=args.n_channels, hidden_channels=args.hidden_size)
            self.seed = torch.nn.functional.pad(make_seed(args.size, args.n_channels), (p, p, p, p), "constant", 0)
            self.pad_target = torch.nn.functional.pad(self.target_img, (p, p, p, p), "constant", 0)
            self.pad_target = self.pad_target.repeat(1, 1, 1, 1)

        # whidden = torch.concat((self.pad_target, torch.zeros((self.size,self.size,12))), axis=2)
        # self.batch_target = tt(np.repeat(whidden[None, ...], 1, 0))

        if args.load_model_path != "":
            self.load_model(args.load_model_path)

    def load_model(self, path):
        """Load a PyTorch model from path."""
        self.net.load_state_dict(torch.load(path))
        if self.es: self.net.double()


    # Do damage on model using pygame, cannot run through ssh
    def interactive(self):

        if self.es: x_eval = tt(np.repeat(self.seed[None, ...], 1, 0))
        else: x_eval = self.seed.clone()

        cellsize = 20
        imgpath = '%s/one.png' % (self.logdir)
        
        pygame.init()
        surface = pygame.display.set_mode((self.size * cellsize, self.size * cellsize))
        pygame.display.set_caption("Interactive CA-ES")

        damaged = 100
        counter = 0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # damage
                    if damaged == 100:
                        dmg_size = 20
                        mpos_x, mpos_y = event.pos
                        mpos_x, mpos_y = mpos_x // cellsize, mpos_y // cellsize
                        # mpos_y = (self.size // 2) + 1
                        # mpos_x = 0
                        dmg_size = self.size
                        if self.es: x_eval[:, mpos_y:mpos_y + dmg_size, mpos_x:mpos_x + dmg_size, :] = 0
                        else:       x_eval[:, :, mpos_y:mpos_y + dmg_size, mpos_x:mpos_x + dmg_size] = 0
                        # damaged = 0 # number of steps to record loss after damage has occurred
                        
                        # # For noise:
                        # l_func = torch.nn.MSELoss()
                        # e = x_eval.detach().cpu()
                        # e.requires_grad = True
                        # l = l_func(e, self.batch_target)
                        # self.net.zero_grad()
                        # l.backward()
                        # x_eval = adv_attack(x_eval, self.eps, e.grad.data)
                        pygame.display.set_caption("Saving loss...")

            x_eval = self.net(x_eval)
            if self.es: image = to_rgb(x_eval).permute(0, 3, 1, 2)
            else: image = to_rgb_ad(x_eval[:, :4].detach().cpu())
            
            save_image(image, imgpath, nrow=1, padding=0)
            
            # Damage at 51:
            # if counter == 51:
            #     # For lower half:
            #     mpos_y = (self.size // 2) + 1
            #     mpos_x = 0
            #     dmg_size = self.size
            #     if self.es: x_eval[:, mpos_y:mpos_y + dmg_size, mpos_x:mpos_x + dmg_size, :] = 0
            #     else:       x_eval[:, :, mpos_y:mpos_y + dmg_size, mpos_x:mpos_x + dmg_size] = 0
            if counter == 400: pygame.quit()
            counter += 1

            loss = self.net.loss(x_eval, self.pad_target)
            self.writer.add_scalar("train/fit", loss, counter)
            
            # # For manual damage:
            # if damaged < 100:
            #     loss = self.net.loss(x_eval, self.pad_target)
            #     self.writer.add_scalar("train/fit", loss, damaged)

            #     if damaged == 99:
            #         pygame.display.set_caption("Interactive CA-ES")
            #     damaged += 1

            # Saving and loading each image as a quick hack to get rid of the batch dimension in tensor
            image = np.asarray(Image.open(imgpath))

            self.game_update(surface, image, cellsize)
            time.sleep(0.05)
            pygame.display.update()

    def game_update(self, surface, cur_img, sz):
        nxt = np.zeros((cur_img.shape[0], cur_img.shape[1]))

        for r, c, _ in np.ndindex(cur_img.shape):
            pygame.draw.rect(surface, cur_img[r,c], (c*sz, r*sz, sz, sz))

        return nxt

def make_seed(size, n_channels):
    x = torch.zeros((1, n_channels, size, size), dtype=torch.float32)
    x[:, 3:, size // 2, size // 2] = 1
    return x

def to_rgb_ad(img_rgba):
    rgb, a = img_rgba[:, :3, ...], torch.clamp(img_rgba[:, 3:, ...], 0, 1)
    return torch.clamp(1.0 - a + rgb, 0, 1)
        