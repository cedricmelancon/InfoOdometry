#!/bin/bash
set -e

pip install -r /main/requirements.txt

python main.py --gpu 0 --dataset mit --epoch 300 --on_the_fly --transition_model deepvio $@