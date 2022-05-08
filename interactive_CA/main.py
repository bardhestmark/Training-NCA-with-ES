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

adam_nonsample_models = [
    ['final_models\\Adam\\NonSamplePools\\9-CARROT-train_05-05-2022_17-55-17\\models\\model_19500.pt', '🥕', 9, False],
    ['final_models\\Adam\\NonSamplePools\\9-RABBIT FACE-train_05-05-2022_16-07-21\\models\\model_19500.pt', '🐰', 9, False],
    ['final_models\\Adam\\NonSamplePools\\15-CARROT-train_05-05-2022_17-31-03\\models\\model_19500.pt', '🥕', 15, False],
    ['final_models\\Adam\\NonSamplePools\\15-RABBIT-FACE-train_05-05-2022_14-39-24\\models\\model_19500.pt', '🐰', 15, False]
]

adam_sample_models = [
    ['final_models\\Adam\\SamplePools\\9-CARROT-train_05-05-2022_17-04-48\\models\\model_19500.pt', '🥕', 9, False],
    ['final_models\\Adam\\SamplePools\\9-RABBIT-FACE-train_05-05-2022_11-43-14\\models\\model_19500.pt', '🐰', 9, False],
    ['final_models\\Adam\\SamplePools\\15-CARROT-train_05-05-2022_17-00-31\\models\\model_19500.pt', '🥕', 15, False],
    ['final_models\\Adam\\SamplePools\\15-RABBIT-FACE-train_05-05-2022_13-44-43\\models\\model_19500.pt', '🐰', 15, False]
]

es_nonsample_models = [
    ['final_models\\ES\\NonSamplePools\\9-CARROT-train_05-05-2022_09-14-06\\models\\model_2212000', '🥕', 9, True],
    ['final_models\\ES\\NonSamplePools\\9-RABBIT FACE-train_06-05-2022_12-18-56\\models\\model_1999000', '🐰', 9, True ],
    ['final_models\\ES\\NonSamplePools\\15-CARROT-train_06-05-2022_11-06-58\\models\\model_1999000', '🥕', 15, True]

]

es_sample_models = [
    ['final_models\\ES\\SamplePools\\9-CARROT-train_29-04-2022_11-33-16\\models\\model_1036000', '🥕', 9, True],
    ['final_models\\ES\\SamplePools\\9-RABBIT FACE-train_01-05-2022_10-32-24\\models\\model_1157000', '🐰', 9, True],
    ['final_models\\ES\\SamplePools\\15-CARROT-train_29-04-2022_11-18-06\\models\\model_1105000', '🥕', 15, True],
    ['final_models\\ES\\SamplePools\\15-RABBIT FACE-train_01-05-2022_10-32-40\\models\\model_1126000', '🐰', 15, True]
]

models = [adam_nonsample_models, adam_sample_models,
          es_nonsample_models, es_sample_models]

if __name__ == '__main__':
    # pick model [Index of model types][index of 9x9 or 15x15 rabbit or carrot]
    model = models[3][3]  # change only this or size

    # Auto determined
    load_model = model[0]
    emoji = model[1]
    emoji_size = model[2]  # size of training image usually 9 or 15
    es = model[3]

    # canvas size
    size = emoji_size

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--img", type=str, default=emoji,
                        metavar="🐰", help="The emoji to train on")
    parser.add_argument("-s", "--size", type=int,
                        default=size, help="Image size")
    parser.add_argument("--logdir", type=str, default="interactive_CA/logs",
                        help="Logging folder for new model")
    parser.add_argument("-l", "--load_model_path", type=str,
                        default=load_model, help="Path to pre trained model")
    parser.add_argument("--n_channels", type=int, default=16,
                        help="Number of channels of the input tensor")
    parser.add_argument("--hidden_size", type=int, default=32,
                        help="Number of hidden channels")
    parser.add_argument("--fire_rate", type=float, default=0.5,
                        metavar=0.5, help="Cell fire rate")
    parser.add_argument("--es", type=bool, default=es,
                        metavar=True, help="ES or adam")
    parser.add_argument("--eps", type=float, default=0.007,
                        help="Epsilon scales the amount of damage done from adversarial attacks")

    args = parser.parse_args()
    args.emoji_size = emoji_size

    if not os.path.isdir(args.logdir):
        raise Exception(
            "Logging directory '%s' not found in base folder" % args.logdir)

    match es:
        case True: method = 'ES'
        case False: method = 'ADAM'

    args.logdir = "%s/%s-%s-%s_%s" % (args.logdir, emoji_size, method,
                                      unicodedata.name(args.img), time.strftime("%d-%m-%Y_%H-%M-%S"))
    os.mkdir(args.logdir)

    logging.basicConfig(filename='%s/logfile.log' %
                        args.logdir, encoding='utf-8', level=logging.INFO)
    argprint = "\nArguments:\n"
    for arg, value in vars(args).items():
        argprint += ("%s: %r\n" % (arg, value))
    logging.info(argprint)

    Interactive = Interactive(args)

    Interactive.interactive()
