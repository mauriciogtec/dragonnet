#! /usr/bin/bash
sousrce ~/.bashrc
conda activate cuda116
export l2=0.01
python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=0 --l2=${l2} --lr 0.001 --device 0 --silent --nseeds=200
python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=1 --l2=${l2} --lr 0.001 --device 0 --silent --nseeds=200
python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=2 --l2=${l2} --lr 0.001 --device 0 --silent --nseeds=200
python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=3 --l2=${l2} --lr 0.001 --device 0 --silent --nseeds=200
python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=4 --l2=${l2} --lr 0.001 --device 0 --silent --nseeds=200
python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=5 --l2=${l2} --lr 0.001 --device 0 --silent --nseeds=200
python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=6 --l2=${l2} --lr 0.001 --device 0 --silent --nseeds=200
