#!/bin/sh
#
# PROGRAMMER: Jeffrey Stocker
#
python train.py --arch mobilenet_v2 --device cuda --learning_rate 0.003 --n_hidden_layers "725, 530, 240, 200" --epochs 20 --images_path lowers/train --images_test_path flowers/test