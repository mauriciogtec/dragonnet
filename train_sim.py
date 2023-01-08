import yaml
import numpy as np
import torch
import os
from datagen import set_seed, sim_data
from dragonnet import DragonNet
import argparse
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset


def run(
    scenario=0, P=100, n=None, nseeds=1, global_seed=12345, batch=64, sgld=False, burnin=0.5, rdir="results",  verbose=True, **kwargs  # args to pass to DragonNet
):
    set_seed(global_seed)
    simulation_seeds = np.random.randint(0, int(1e6), size=nseeds)

    for seed in simulation_seeds:
        set_seed(seed)
        # simulate data
        X, A, Y_0, Y_1, Y, scaler = sim_data(scenario, n=300 if scenario != 4 else 500)
        n, P = X.shape
        epochs = 1000 if sgld else 500

        true_ate = scaler.scale_[0] * (Y_1 - Y_0).mean()
        X, A, Y = [torch.FloatTensor(u) for u in (X, A, Y)]
        dataset = TensorDataset(X, A, Y)
        dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, pin_memory=True, persistent_workers=True, num_workers=8)

        # create dragonnet model
        nbatches = (n // batch)  # nbatches used for burnin period and adjust SGLD
        burnin = int(nbatches * epochs * burnin)  # burnin length, only for SGLD
        buffer_size = epochs // 4 if sgld else 1
        sampling_kwargs = dict(num_pseudo_batches=nbatches, burnin=burnin, metrics_buffer_size=buffer_size)
        model = DragonNet(P, sgld=sgld, **sampling_kwargs, **kwargs)

        # fit model
        trainer = pl.Trainer(accelerator="cpu", devices="auto", max_epochs=epochs, logger=[], enable_checkpointing=False, gradient_clip_val=10.0, enable_progress_bar=verbose)
        trainer.fit(model, dataloader)

        # correct the buffer for the initial normalization of y and add true_ate rror
        model.buffer['ate_estimate'] = scaler.scale_[0] * np.array(model.buffer['ate_estimate']) 
        model.buffer['mse_loss'] = scaler.scale_[0] * np.array(model.buffer['mse_loss'])
        model.buffer['ate_error'] = model.buffer['ate_estimate'] - true_ate

        # save fit results
        results = {}
        qbrks = [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975]
        for metric, estimates in model.buffer.items():
            qvals = np.quantile(estimates, qbrks)
            if sgld:
                results[metric] = dict(
                    **{"q" + str(k): float(v) for k, v in zip(qbrks, qvals)},
                    mean=float(np.mean(estimates)),
                    sd=float(np.std(estimates))
                )
            else:
                results[metric] = float(estimates[0])
        mean_results = {k: v['mean'] for k, v in results.items()} if sgld else results
        print(f"Results: {mean_results}")
        os.makedirs(f"{rdir}", exist_ok=True)
        with open(f"{rdir}/scenario_{scenario}-seed_{seed:06d}{'_sgld' if sgld else ''}.yaml", "w") as io:
            yaml.dump(results, io)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default=None, type=int)
    parser.add_argument("--nseeds", default=None, type=int)
    parser.add_argument("--dbody", default=None, type=int)
    parser.add_argument("--dhead", default=None, type=int)
    parser.add_argument("--depth", default=None, type=int)
    parser.add_argument("--n", default=None, type=int)
    parser.add_argument("--P", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--burnin", default=None, type=float)
    parser.add_argument("--batch", default=None, type=int)
    parser.add_argument("--sgld", default=None, action="store_true")
    parser.add_argument("--l2", default=None, type=float)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--silent", default=None, dest="verbose", action="store_false")
    parser.add_argument("--rdir", default=None, type=str)
    args = parser.parse_args()
    run(**{k: v for k,v in vars(args).items() if v is not None})
