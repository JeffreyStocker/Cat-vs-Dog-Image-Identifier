#!/bin/sh
#
# PROGRAMMER: Jeffrey Stocker
#
python pedict.py --device cuda --topk 5 --class_values cat_to_name.json --image_path flowers/test/2/image_05100.jpg --checkpoint checkpoint/densenet121_checkpoints_2020-04-30 10-44-07.027571-e173.pth