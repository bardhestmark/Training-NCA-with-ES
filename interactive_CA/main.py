import time
import logging
import argparse
import time
import os
import unicodedata
import logging

from interactive import Interactive

# rabbit 🐰
# carrot 🥕
# watermelon 🍉
# lizard 🦎

models = [
    'final_models\9-CARROT-train_29-04-2022_11-33-16\models\model_1035000',
    'final_models\\15-CARROT-train_29-04-2022_11-18-06\models\model_1104000',
    'automata_ex\logs\CARROT-train_01-05-2022_14-19-47\models\model_16000.pt',
    'automata_ex\logs\CARROT-train_01-05-2022_15-08-31\models\model_79000.pt',
    'CA-ES\saved_models\\20_lizard'
]

if __name__ == '__main__':
    emoji = '🦎'
    load_model = models[4]
    size = 20
    es = True

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img", type=str, default=emoji, metavar="🐰", help="The emoji to train on")
    parser.add_argument("-s", "--size", type=int, default=size, help="Image size")
    parser.add_argument("--padding", type=int, default=0, help="Padding. The shape after padding is (h + 2 * p, w + 2 * p).")
    parser.add_argument("--logdir", type=str, default="interactive_CA/logs", help="Logging folder for new model")
    parser.add_argument("-l", "--load_model_path", type=str, default=load_model, help="Path to pre trained model")
    parser.add_argument("--n_channels", type=int, default=16, help="Number of channels of the input tensor")
    parser.add_argument("--hidden_size", type=int, default=32, help="Number of hidden channels")
    parser.add_argument("--fire_rate", type=float, default=0.5, metavar=0.5, help="Cell fire rate")
    parser.add_argument("--es", type=bool, default=es, metavar=True, help="ES or adam")
    parser.add_argument("--eps", type=float, default=0.007, help="Epsilon scales the amount of damage done from adversarial attacks")

    args = parser.parse_args()

    if not os.path.isdir(args.logdir):
        raise Exception("Logging directory '%s' not found in base folder" % args.logdir)

    args.logdir = "%s/ES-%s_%s" % (args.logdir, unicodedata.name(args.img), time.strftime("%d-%m-%Y_%H-%M-%S"))
    os.mkdir(args.logdir)

    logging.basicConfig(filename='%s/logfile.log' % args.logdir, encoding='utf-8', level=logging.INFO)
    argprint = "\nArguments:\n"
    for arg, value in vars(args).items():
        argprint += ("%s: %r\n" % (arg, value))
    logging.info(argprint)

    Interactive = Interactive(args)

    Interactive.interactive()


