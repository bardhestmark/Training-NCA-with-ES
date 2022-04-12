import argparse
import os
import unicodedata
import logging
from damage import Damage
import time
if __name__ == '__main__':

    emoji = '🥕'

    parser = argparse.ArgumentParser()

    parser.add_argument("--n_iterations", type=int, default=10000, help="Number of iterations to test for.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--dmg_freq", type=int, default=500, help="Frequency for damaging",)

    parser.add_argument("--alpha", type=float, default=0.005, metavar=0.005, help="Alpha for how much noise to add")
    parser.add_argument("--padding", type=int, default=0, help="Padding for image")
    
    parser.add_argument("--img", type=str, default=emoji, metavar="🐰", help="The emoji to train on")
    parser.add_argument("--size",type=int, default=15, help="size of image")
    parser.add_argument("--logdir", type=str, default="logs", help="Logging folder for new model")
    parser.add_argument("--load_model_path", type=str, default="models/model_999500", help="Path to pre trained model")

    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        raise Exception("Logging directory '%s' not found in base folder" % args.logdir)

    args.logdir = "%s/%s_%s" % (args.logdir, unicodedata.name(args.img), time.strftime("%d-%m-%Y_%H-%M-%S"))
    os.mkdir(args.logdir)

    print(args.logdir)
    logging.basicConfig(handlers=[logging.FileHandler(filename=f'{args.logdir}/logfile.log', encoding='utf-8', mode='a+')], level=logging.INFO)
    argprint = "\nArguments:\n"
    for arg, value in vars(args).items():
        argprint += ("%s: %r\n" % (arg, value))
    logging.info(argprint)

    #dmg here
    dmg = Damage(args)
    dmg.run()
    