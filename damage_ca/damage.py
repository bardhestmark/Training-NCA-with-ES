import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from model import CellularAutomataModel
from utils import load_emoji, to_rgb, get_gaussian_kernel
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
        self.mode = args.mode # 0 for blur, 1 for pixel removal, 2 for adversarial attck

        p = self.padding
        self.pad_target = F.pad(tt(self.target_img), (0, 0, p, p, p, p))
        h, w = self.pad_target.shape[:2]
        self.seed = np.zeros([h, w, 16], np.float64)
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
        x_eval = self.x0.clone()
        eval_video = torch.empty(1, self.n_iterations, 3, self.size, self.size)
        blur = get_gaussian_kernel(channels=16)
        for i in range(self.n_iterations):
            x_eval_out = to_rgb(x_eval).permute(0, 3, 1, 2)
            eval_video[0, i] = x_eval_out
            loss=self.net.loss(x_eval, self.pad_target)
            self.writer.add_scalar("dmg/loss", loss, i)

            x_eval = self.net(x_eval) #update step
            
            if i % self.dmg_freq == 0: # do damage
                if self.mode == 0:
                    gblur = blur(x_eval[0].permute(2, 0, 1).type(torch.float32)) # returns (ch, s, s)
                    x_eval[0] = gblur.permute(1,2,0)
                    #self.writer.add_image(f'after_{i}', to_rgb(x_eval)[0].permute(2, 0, 1))
                elif self.mode == 1:
                    #lower half
                    y_pos = (self.size // 2) + 1
                    x_pos = 0
                    dmg_size = self.size
                    x_eval[:, y_pos:y_pos + dmg_size, x_pos:x_pos + dmg_size, :] = 0

                image = to_rgb(x_eval).permute(0, 3, 1, 2)
                save_image(image, imgpath, nrow=1, padding=0)

        imgpath = '%s/done.png' % (self.logdir)
        image = to_rgb(x_eval).permute(0, 3, 1, 2)
        save_image(image, imgpath, nrow=1, padding=0)
        self.writer.add_video("eval_damage", eval_video, 100, fps=60)
        
                    
    