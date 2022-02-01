Based on video 'Growing neural cellular automata in PyTorch': https://www.youtube.com/watch?v=21ACbWoF2Oo

Everything in this folder is from https://github.com/jankrepl/mildlyoverfitted/tree/master/github_adventures/automata
The project had no licence

Install package moviepy to env

Run tensorboard with 
``
tensorboard --logdir=logs
``

Get parameters:
``python train.py --help``

Run args suggestion:
``python train.py -d cuda -n 10000 -b 4 rabbit.png``
Remove '-d cuda' to train using cpu

In pytorch edit configuration and add parameters:
``-d cuda -n 10000 -b 4 rabbit.png``

