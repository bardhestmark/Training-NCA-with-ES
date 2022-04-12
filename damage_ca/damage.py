import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from model import CellularAutomataModel
from utils import load_emoji, to_rgb
import torch.nn.functional as F
from torch import tensor as tt
import numpy as np

class Damage():
    def __init__(self, args):
        self.n_iterations = args.n_iterations
        self.batch_size = args.batch_size
        self.dmg_freq = args.dmg_freq
        self.alpha = args.alpha
        self.padding = args.padding
        self.size = args.size+2 +self.padding
        self.logdir = args.logdir
        self.load_model_path = args.load_model_path
        self.n_channels = args.n_channels
        self.target_img = load_emoji(args.img, self.size) #rgba img

        p = self.padding
        self.pad_target = F.pad(tt(self.target_img), (0, 0, p, p, p, p))
        h, w = self.pad_target.shape[:2]
        self.seed = np.zeros([h, w, self.n_channels], np.float64)
        self.seed[h // 2, w // 2, 3:] = 1.0
        self.x0 = tt(np.repeat(self.seed[None, ...], self.batch_size, 0)) #seed

        t_rgb = to_rgb(self.pad_target).permute(2, 0, 1)
        self.net = CellularAutomataModel(n_channels=16, fire_rate=0.5, hidden_channels=32)
        save_image(t_rgb, "%s/target_image.png" % self.logdir)
        self.writer = SummaryWriter(self.logdir)

        
    def load_model(self, path):
        """Load a PyTorch model from path."""
        self.net.load_state_dict(torch.load(path))
        self.net.double()

    def run(self):
        imgpath = '%s/damaged.png' % (self.logdir)
        self.load_model(self.load_model_path) # model loaded
        x = self.x0.clone()
        for i in range(self.n_iterations): # fully grow first
            x_eval = self.net(x)
            loss = self.net.loss(x_eval, self.pad_target)
            self.writer.add_scalar("dmg/loss", loss, i)

            if i % self.dmg_freq == 0: # do damage
                #lower half
                y_pos = (self.size // 2) + 1
                dmg_size = self.size
                x_eval[:, y_pos:y_pos + dmg_size, 0:0 + dmg_size, :] = 0
                image = to_rgb(x_eval).permute(0, 3, 1, 2)
                save_image(image, imgpath, nrow=1, padding=0)
        
        imgpath = '%s/done.png' % (self.logdir)
        image = to_rgb(x_eval).permute(0, 3, 1, 2)
        save_image(image, imgpath, nrow=1, padding=0)
                    
    