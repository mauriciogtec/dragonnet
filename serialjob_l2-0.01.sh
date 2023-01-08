#! /usr/bin/bash
# source ~/.bashrc
conda activate torchmac
export l2=0.01

for s in {0..6}
do 
    python train_sim.py --rdir=results_lr-0.001-l2-${l2} --scenario=$s --l2=${l2} --lr 0.001 --silent --nseeds=200 &
done
wait

for s in {0..19}
do 
    python train_app.py --seed $s --silent
done
wait
